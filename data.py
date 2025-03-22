import numpy as np
import torch
from torch import nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from estimator import inverse_map

def generate_multivariate_distributions(cluster_centers, label_list, covariances, sample_size=100, seed=None, to_torch_dataset=False):
    """
    Generate multivariate distributions.
    
    Parameters:
    cluster_centers (list): List of cluster centers.
    covariances (list): List of covariance matrices for each cluster.
    sample_size (int): Number of samples per cluster.
    seed (int, optional): Random seed for reproducibility.
    to_torch_dataset (bool, optional): If True, transform the return to torch.utils.data.TensorDataset.
    
    Returns:
    list or TensorDataset: List of generated multivariate distributions or TensorDataset.
    """
    if seed is not None:
        np.random.seed(seed)
    coordinates = []
    labels = []
    for i, center in enumerate(cluster_centers):
        cov = covariances[i]
        distribution = np.random.multivariate_normal(center, cov, sample_size)  # Generate samples per cluster
        coordinates.append(distribution)
        labels.append(np.full(sample_size, label_list[i]))
    
    coordinates = np.vstack(coordinates)
    labels = np.hstack(labels)
    
    if to_torch_dataset:
        dataset = TensorDataset(torch.tensor(coordinates, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
        return dataset
    
    return coordinates, labels

def plot_multivariate_distributions(distributions):
    """
    Plot multivariate distributions.
    
    Parameters:
    distributions (tuple or TensorDataset): Tuple of coordinates and labels or TensorDataset.
    """
    if isinstance(distributions, TensorDataset):
        data = distributions.tensors[0].numpy()
        labels = distributions.tensors[1].numpy()
    else:
        data, labels = distributions
    
    unique_labels = np.unique(labels)
    total_points = len(data)
    point_size = max(1, 1000 // total_points)  # Adjust point size based on the number of points
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        plt.scatter(data[labels == label, 0], data[labels == label, 1], label=f'Cluster {label}', color=colors[i], alpha=0.6, s=point_size)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.title('Multivariate Distributions')
    plt.show()

def transition_labels(labels, transition_prob_matrix, label_list=None):
    """
    Transition labels with a transition probability matrix.
    if label_list is None, it will be inferred from the unique labels in the labels with sorted order.
    
    Parameters:
    labels (np.array): Original labels.
    transition_prob_matrix (np.array): Transition probability matrix.
    
    Returns:
    np.array: Transformed labels.
    """
    if type(labels) is torch.Tensor:
        labels = labels.detach().numpy()
    if type(transition_prob_matrix) is torch.Tensor:
        transition_prob_matrix = transition_prob_matrix.detach().numpy()
    if type(label_list) is torch.Tensor:
        label_list = label_list.detach().numpy()
    if type(labels) is list:
        labels = np.array(labels)
    if type(transition_prob_matrix) is list:
        transition_prob_matrix = np.array(transition_prob_matrix)
    if type(label_list) is list:
        label_list = np.array(label_list)
    new_labels = labels.copy()
    if label_list is None:
        label_list = np.unique(labels)

    if type(label_list) is list:
        label_list = np.array(label_list)
    if type(label_list) is torch.Tensor:
        label_list = label_list.detach().numpy()
    for i, label in enumerate(labels):
        # if label_list is not None:
        new_labels[i] = np.random.choice(label_list, p=transition_prob_matrix[:, inverse_map(label_list, label)].flatten())
        # new_labels[i] = np.random.choice(transition_prob_matrix.shape[0], p=transition_prob_matrix[:, label])
    return new_labels

def placeholder_function():
    """
    This is a placeholder function.
    Replace this with actual implementation.
    """
    pass

# generate_multivariate_distributions([np.array([1, 1]), np.array([5, 5]), np.array([9, 9])], 2, [np.eye(2) for _ in range(3)], 1000, 42)