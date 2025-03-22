import numpy as np
import torch
from torch import nn

def inverse_map(array, values):
    """
    Map from a np.array values to the indices.
    
    Parameters:
    array (np.array): Input array.
    values (np.array or int or float): Values to find in the array.
    
    Returns:
    np.array: Indices of the values in the array.
    """
    if type(array) is torch.Tensor:
        array = array.detach().numpy()
    if type(values) is torch.Tensor:
        values = values.detach().numpy()
    if type(array) is list:
        array = np.array(array)
    if type(values) is list:
        values = np.array(values)
    if isinstance(values, (int, np.integer, float, np.floating)):
        values = np.array([values])
    return np.array([np.where(array == value)[0][0] for value in values])

class HingeLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(HingeLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Manually written hinge loss function.
        
        Parameters:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.
        
        Returns:
        torch.Tensor: Hinge loss.
        """
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        loss = torch.clamp(1 - y_true * y_pred, min=0)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class SigmoidLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(SigmoidLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Manually written sigmoid loss function.
        
        Parameters:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.
        
        Returns:
        torch.Tensor: Sigmoid loss.
        """
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        loss = 1 / (1 + torch.exp(y_true * y_pred))
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

# 0-1 loss
class NELoss(nn.Module):
    def __init__(self, reduction='none'):
        super(NELoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Manually written 0-1 loss function.
        
        Parameters:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.
        
        Returns:
        torch.Tensor: 0-1 loss.
        """
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        loss = - torch.sign(y_pred*y_true) / 2 + 0.5
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

@torch.no_grad()
def calculate_posterior_prob(prob_matrix, prior_prob):
    """
    Calculate posterior probability from a probability matrix using log computation.
    
    Parameters:
    prob_matrix (torch.Tensor): Probability matrix.
    P(Y_1|X_1), P(Y_1|X_2), P(Y_1|X_3)
    P(Y_2|X_1), P(Y_2|X_2), P(Y_2|X_3)
    P(Y_3|X_1), P(Y_3|X_2), P(Y_3|X_3)
    
    Returns:
    torch.Tensor: Posterior probabilities.
    P(X_1|Y_1), P(X_1|Y_2), P(X_1|Y_3)
    P(X_2|Y_1), P(X_2|Y_2), P(X_2|Y_3)
    P(X_3|Y_1), P(X_3|Y_2), P(X_3|Y_3)
    """
    prior_prob = prior_prob.flatten()
    log_prob_matrix = torch.log(prob_matrix)
    log_prior_prob = torch.log(prior_prob)
    log_joint_prob = torch.add(log_prob_matrix, log_prior_prob)
    # print(log_joint_prob)
    log_sum = torch.logsumexp(log_joint_prob, dim=1, keepdim=True)
    # print(log_sum.exp())
    # print(log_sum.shape, log_prob_matrix.transpose(0, 1).shape)
    posterior_prob = torch.exp(torch.sub(log_joint_prob, log_sum)).transpose(0, 1)
    return posterior_prob

class BayesianLoss(nn.Module):
    def __init__(self, optimizer, prob_matrix, prior_prob=None, label_list=None, reduction='none'):
        super(BayesianLoss, self).__init__()
        if type(prob_matrix) is list:
            prob_matrix = torch.tensor(prob_matrix, dtype=torch.float32)
        elif type(prob_matrix) is np.ndarray:
            prob_matrix = torch.from_numpy(prob_matrix, dtype=torch.float32)
        if prior_prob is None:
            prior_prob = torch.ones(prob_matrix.shape[1]) / prob_matrix.shape[1]
        if label_list is None:
            label_list = torch.arange(prob_matrix.shape[1])
        self.label_list = label_list
        self.prior_prob = prior_prob
        self.posterior_prob = calculate_posterior_prob(prob_matrix, prior_prob)
        self.optimizer = optimizer
        self.reduction = reduction
    def forward(self, output, target):
        # print(torch.full(target.shape, self.labels[0]))
        # print([self.optimizer(output, torch.full(target.shape, label)) for label in self.labels])
        loss_list = torch.vstack([self.optimizer(output, torch.full(target.shape, label)) for label in self.label_list])
        loss = torch.zeros(target.shape)
        # print(target.int().detach().numpy())
        for i, l in enumerate(loss_list):
            # print(loss, self.posterior_prob[i, inverse_map(self.label_list, target.int().detach().numpy())] * loss_list[i])
            loss += self.posterior_prob[i, inverse_map(self.label_list, target.int().detach().numpy())] * l
        # loss /= self.label_list.numel()
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class NagarajanLoss(nn.Module):
    '''
    Implementation of the loss function proposed by Nagarajan et al. (2013).
    Only for binary classification.
    [(1-P(flip|Y=-y))l(output, y)-P(flip|Y=y)l(output, -y)] / (1-P(flip|Y=-y)-P(flip|Y=y))
    '''
    def __init__(self, optimizer, flip_prob, reduction='none'):
        super(NagarajanLoss, self).__init__()
        if flip_prob is None:
            flip_prob = torch.tensor([0, 0, 0])
        elif type(flip_prob) is list:
            flip_prob = torch.tensor(flip_prob)
        elif type(flip_prob) is np.ndarray:
            flip_prob = torch.from_numpy(flip_prob)
        if flip_prob.dim() > 1:
            raise ValueError("flip_prob must be a 1D tensor.")
        if flip_prob.shape[0] == 2:
            # print(flip_prob, flip_prob[0], type(flip_prob[0]), torch.cat((flip_prob, flip_prob[0])))
            flip_prob = torch.cat((flip_prob, flip_prob[0].clone().reshape(1)))
        self.flip_prob = flip_prob
        self.optimizer = optimizer
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        loss = (1-self.flip_prob[-y_true.int().detach().numpy()]) * self.optimizer(y_pred, y_true) - self.flip_prob[y_true.int().detach().numpy()] * self.optimizer(y_pred, -y_true)
        loss /= 1 - self.flip_prob[-1] - self.flip_prob[1]
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

# Example usage
if __name__ == "__main__":
    y_true = torch.tensor([1, -1, 1, -1], dtype=torch.float32)
    y_pred = torch.tensor([0.8, -0.5, 0.3, -0.2], dtype=torch.float32)
    hinge_loss_mean = HingeLoss(reduction='mean')
    hinge_loss_sum = HingeLoss(reduction='sum')
    hinge_loss_none = HingeLoss(reduction='none')
    print(f"Hinge Loss (mean): {hinge_loss_mean(y_true, y_pred).item()}")
    print(f"Hinge Loss (sum): {hinge_loss_sum(y_true, y_pred).item()}")
    print(f"Hinge Loss (none): {hinge_loss_none(y_true, y_pred)}")

    sigmoid_loss_mean = SigmoidLoss(reduction='mean')
    sigmoid_loss_sum = SigmoidLoss(reduction='sum')
    sigmoid_loss_none = SigmoidLoss(reduction='none')
    print(f"Sigmoid Loss (mean): {sigmoid_loss_mean(y_true, y_pred).item()}")
    print(f"Sigmoid Loss (sum): {sigmoid_loss_sum(y_true, y_pred).item()}")
    print(f"Sigmoid Loss (none): {sigmoid_loss_none(y_true, y_pred)}")

    prob_matrix = torch.tensor([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]], dtype=torch.float32).transpose(0, 1)
    prior_prob = torch.tensor([0.3, 0.4, 0.3], dtype=torch.float32)
    posterior_prob = calculate_posterior_prob(prob_matrix, prior_prob)
    print(f"Posterior Probabilities:\n{posterior_prob}")

    # Test Bayesian Estimator
    optimizer = HingeLoss(reduction='none')
    prob_matrix = torch.tensor([[0.2, 0.8], [0.5, 0.5]], dtype=torch.float32).transpose(0, 1)
    bayesian_estimator = BayesianLoss(optimizer, prob_matrix, label_list=torch.tensor([1, -1]), reduction='mean')
    output = torch.tensor([0.5, -0.5], dtype=torch.float32)
    target = torch.tensor([1, -1], dtype=torch.float32)
    loss = bayesian_estimator(output, target)
    print(f"Bayesian Estimator Loss: {loss.item()}")

    # Test inverse_map function
    array = np.array([10, 20, 30, 40, 50])
    values = np.array([30, 10, 50])
    indices = inverse_map(array, values)
    print(f"Indices of {values} in array: {indices}")
