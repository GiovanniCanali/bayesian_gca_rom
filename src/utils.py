from matplotlib.colors import LinearSegmentedColormap
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Custom color map for better visualization
min_val = 0.01
max_val = 0.85
new_colors = plt.cm.twilight(np.linspace(min_val, max_val, 256))
c = LinearSegmentedColormap.from_list("twilight_trimmed", new_colors)

@torch.no_grad()
def plot(pred, target, var_prediction, pos, triang, idx, save_dir):
    """
    Plotter for the solution, error, and predictive variance (if available).
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Flatten the data for plotting
    pred = pred[idx].flatten()
    target = target[idx].flatten()
    x = pos[idx, :, 0].flatten()
    y = pos[idx, :, 1].flatten()
    sq_err = (target - pred).pow(2)

    # Flatten the variance prediction if it exists
    if var_prediction is not None:
        var_prediction = var_prediction[idx].flatten()

    # Determine the color scale limits
    vmin = min(pred.min(), target.min())
    vmax = max(pred.max(), target.max())

    # Update matplotlib parameters for better visualization
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 16 if var_prediction is not None else 12,
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
        }
    )

    # Plot deterministic results
    if var_prediction is None:
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.tricontourf(x, y, triang, target, 100, cmap=c, vmin=vmin, vmax=vmax)
        if "airfoil" in save_dir:
            plt.ylim(-0.75, 0.75)
        plt.title(r"Ground Truth $y$")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.tricontourf(x, y, triang, pred, 100, cmap=c, vmin=vmin, vmax=vmax)
        plt.title(r"Prediction GCA-ROM $\hat{y}$")
        plt.colorbar()
        if "airfoil" in save_dir:
            plt.ylim(-0.75, 0.75)

        plt.subplot(1, 3, 3)
        plt.tricontourf(x, y, triang, sq_err, 100, cmap=c, vmin=vmin, vmax=vmax)
        if "airfoil" in save_dir:
            plt.ylim(-0.75, 0.75)
        plt.title(r"Square Error $|y - \hat{y}|^2$")
        plt.colorbar()

        file_name = f"{save_dir}/solution_{idx}.png"

    # Plot Bayesian results
    else:
        plt.figure(figsize=(16, 12))
        plt.subplot(2, 2, 1)
        plt.title(r"Ground Truth $y$")
        plt.tricontourf(x, y, triang, target, 100, cmap=c, vmin=vmin, vmax=vmax)
        if "airfoil" in save_dir:
            plt.ylim(-0.75, 0.75)
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.tricontourf(x, y, triang, pred, 100, cmap=c, vmin=vmin, vmax=vmax)
        if "airfoil" in save_dir:
            plt.ylim(-0.75, 0.75)
        title = (
            r"Prediction Bayesian GCA-ROM $\hat{y}$"
            if "Bayes" in save_dir
            else r"Prediction Ensemble GCA-ROM $\hat{y}$"
        )
        plt.title(title)
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.title(r"Square Error $|y - \hat{y}|^2$")
        plt.tricontourf(x, y, triang, sq_err, 100, cmap=c, vmin=vmin, vmax=vmax)
        if "airfoil" in save_dir:
            plt.ylim(-0.75, 0.75)
        plt.colorbar()
        plt.ticklabel_format()

        plt.subplot(2, 2, 4)
        plt.title(r"Predictive Variance $\mathrm{Var}(\hat{y})$")
        plt.tricontourf(x, y, triang, var_prediction, 100, cmap=c)
        if "airfoil" in save_dir:
            plt.ylim(-0.75, 0.75)
        plt.colorbar()

        file_name = f"{save_dir}/solution_bayesian_{idx}.png"

    plt.ticklabel_format()
    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    plt.close()


class CustomMSELoss(torch.nn.MSELoss):
    """
    Create a custom MSE loss that can handle both tensors and PyG Data objects.
    """

    def forward(self, output, target):
        """
        Compute the MSE loss.
        """
        if isinstance(output, Data):
            output = output.x
        if isinstance(target, Data):
            target = target.x
        return torch.nn.functional.mse_loss(
            output, target, reduction=self.reduction
        )
