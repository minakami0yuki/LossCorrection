import numpy as np
import torch
import matplotlib.pyplot as plt

def illustrate_decision_boundary(model, dataset, name='Decision Boundary', resolution=100, plot_data=True):
    """
    Illustrate the decision boundary of a model.
    
    Parameters:
    model (nn.Module): The model to illustrate.
    dataset (TensorDataset): The dataset to plot.
    resolution (int, optional): The resolution of the grid. Default is 100.
    plot_data (bool, optional): Whether to plot the data points. Default is True.
    
    Returns:
    None
    """
    # Extract data and labels from the dataset
    data, labels = dataset.tensors
    data = data.numpy()
    labels = labels.numpy()
    
    # Create a grid of points
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict the labels for the grid points
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        predictions = model(grid_tensor).sign().numpy()
        predictions = predictions.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, predictions, alpha=0.8, cmap=plt.cm.PiYG)
        
    # Plot the data points
    if plot_data:
        unique_labels = np.unique(labels)
        total_points = len(data)
        point_size = max(1, 1000 // total_points)  # Adjust point size based on the number of points
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(unique_labels):
            plt.scatter(data[labels == label, 0], data[labels == label, 1], label=f'Class {label}', color=colors[i], s=point_size)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.title(name)
    plt.show()
