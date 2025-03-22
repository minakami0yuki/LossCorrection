import numpy as np
import torch
from torch import nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, TensorDataset
from estimator import inverse_map, NELoss

class OneLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.Identity(), name="OneLayerNN"):
        super(OneLayerNN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.activation = activation
        self.name = name

    def forward(self, x):
        return self.activation(self.linear(x))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

class HyperparameterSpace:
    def __init__(self, learning_rate=[0.1, 0.07, 0.03, 0.01, 0.007, 0.003, 0.001], momentum=[0.99, 0.9, 0.5, 0.3, 0], batch_size=[32, 64, 128], epochs=[16, 32, 64]):
        self.learning_rate = np.array(learning_rate)
        self.momentum = np.array(momentum)
        self.batch_size = np.array(batch_size)
        self.__index__ = np.array([0, 0, 0])

    def __iter__(self):
        self.reset_index()
        return self

    def get_index(self):
        return self.__index__
    
    def reset_index(self):
        self.__index__ = np.array([0, 0, 0])

    def __len__(self):
        return len(self.learning_rate) * len(self.momentum) * len(self.epochs)
    
    def __getitem__(self, index):
        return (self.learning_rate[index[0]].item(), self.momentum[index[1]].item(), self.batch_size[index[2]].item())

    def __next__(self):
        if self.__index__[0] == len(self.learning_rate):
            raise StopIteration
        item = self.__getitem__(self.__index__)
        self.__index__[2] += 1
        if self.__index__[2] == len(self.batch_size):
            self.__index__[2] = 0
            self.__index__[1] += 1
        if self.__index__[1] == len(self.momentum):
            self.__index__[1] = 0
            self.__index__[0] += 1
        return item

def scale_loss(loss, batch_size, total, reduction):
    if reduction == 'mean':
        loss *= batch_size
    elif reduction == 'none':
        return loss.sum()
    elif reduction == 'sum':
        pass
    loss /= total
    # print(loss)
    return loss

class ModelPact():
    def __init__(self, model, optimizer, criterion, benchmark, learning_rate, momentum, batch_size, epochs=50, plot_loss=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.benchmark = benchmark
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.epochs = epochs
        self.plot_loss = plot_loss

    def train(self, dataset):
        return train(self.model, dataset, self.optimizer, self.criterion, self.benchmark, self.epochs, self.batch_size, plot_loss=self.plot_loss)
    
    def evaluate(self, dataset):
        return evaluate(self.model, dataset, self.criterion, self.benchmark, self.batch_size)

def train(model, dataset, optimizer, criterion, benchmark, epochs=20, batch_size=32, val_split=0.2, plot_loss=False):
    """
    Train the model on a dataset with a given optimizer.
    
    Parameters:
    model (nn.Module): The model to train.
    dataset (TensorDataset or tuple): The dataset to train on.
    optimizer (torch.optim.Optimizer): The optimizer to use.
    criterion (nn.Module): The loss function to use.
    benchmark (nn.Module): The benchmark function to measure performance.
    epochs (int, optional): Number of epochs to train. Default is 20.
    batch_size (int, optional): Batch size for training. Default is 32.
    val_split (float, optional): Ratio of the dataset to use for validation. Default is 0.2.
    plot_loss (bool, optional): Whether to plot training loss on the val set. Default is False.
    
    Returns:
    None
    """
    # Transform dataset into TensorDataset if it is not already
    if not isinstance(dataset, TensorDataset):
        data, labels = dataset
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    
    # Split the dataset into train and val sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    val_losses = []
    val_benchmarks = []
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        val_benchmark = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                benchmark_value = benchmark(outputs, targets)
                val_benchmark += benchmark_value.item()
        
        # print(val_loss, val_benchmark, val_size)
        val_loss = scale_loss(val_loss, batch_size, val_size, criterion.reduction)
        val_benchmark = scale_loss(val_benchmark, batch_size, val_size, benchmark.reduction)
        
        val_losses.append(val_loss)
        val_benchmarks.append(val_benchmark)
        
        if plot_loss == True:
            print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Benchmark: {val_benchmark:.4f}")
        
        
    if plot_loss:
        plt.figure(figsize=(10, 5))
        plt.plot(val_losses, label='Val Loss')
        plt.plot(val_benchmarks, label='Val Benchmark')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Validation Loss and Benchmark')
        plt.show()
    return val_losses, val_benchmarks

def evaluate(model, dataset, criterion, benchmark, batch_size=32):
    """
    Evaluate the model on a dataset.
    
    Parameters:
    model (nn.Module): The model to evaluate.
    dataset (TensorDataset or tuple): The dataset to evaluate on.
    criterion (nn.Module): The loss function to use.
    benchmark (nn.Module): The benchmark function to measure performance.
    batch_size (int, optional): Batch size for evaluation. Default is 32.
    
    Returns:
    tuple: Evaluation loss and benchmark value.
    """
    # Transform dataset into TensorDataset if it is not already
    if not isinstance(dataset, TensorDataset):
        data, labels = dataset
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    eval_loss = 0.0
    eval_benchmark = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            eval_loss += loss.item()
            benchmark_value = benchmark(outputs, targets)
            eval_benchmark += benchmark_value.item()
            total += inputs.size(0)
    
    eval_loss = scale_loss(eval_loss, batch_size, total, criterion.reduction)
    eval_benchmark = scale_loss(eval_benchmark, batch_size, total, benchmark.reduction)
    
    return eval_loss, eval_benchmark

# Example usage
if __name__ == "__main__":
    input_dim = 2
    output_dim = 1
    model = OneLayerNN(input_dim, output_dim)
    print(model)

    # Test HyperparameterSpace
    hyperparameter_space = HyperparameterSpace()
    print(f"Total combinations: {len(hyperparameter_space)}")
    for params in hyperparameter_space:
        print(params)
