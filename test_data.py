import numpy as np
import matplotlib.pyplot as plt
from data import generate_multivariate_distributions, plot_multivariate_distributions
import os
import torch
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def test_generate_multivariate_distributions():
    cluster_centers = [np.array([1, 1]), np.array([5, 5]), np.array([9, 9])]
    dimensions = 2
    covariances = [np.eye(dimensions) for _ in range(len(cluster_centers))]
    sample_size = 1000
    seed = 42

    distributions = generate_multivariate_distributions(cluster_centers, dimensions, covariances, sample_size, seed)
    assert len(distributions) == len(cluster_centers)
    assert all(dist.shape == (sample_size, dimensions) for dist in distributions)
    print("generate_multivariate_distributions passed.")

def test_plot_multivariate_distributions():
    cluster_centers = [np.array([1, 1]), np.array([5, 5]), np.array([9, 9])]
    dimensions = 2
    covariances = [np.eye(dimensions) for _ in range(len(cluster_centers))]
    sample_size = 1000
    seed = 42

    distributions = generate_multivariate_distributions(cluster_centers, dimensions, covariances, sample_size, seed)
    plot_multivariate_distributions(distributions)
    print("plot_multivariate_distributions passed.")

if __name__ == "__main__":
    # test_generate_multivariate_distributions()
    # test_plot_multivariate_distributions()
    tensor1 = torch.randn(2, 2, 2)
    tensor2 = torch.randn(2, 1)
    torch.matmul(tensor1, tensor2).size()
    print(tensor1, tensor2, torch.matmul(tensor1, tensor2))
