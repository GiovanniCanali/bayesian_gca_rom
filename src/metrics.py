from scipy.stats import spearmanr
import csv, os
import torch


class Metrics:
    """
    Class to compute the metrics for evaluating the performance of a model
    that predicts a target solution with associated uncertainty.
    """

    def __init__(
        self,
        target_solution,
        mean_prediction,
        var_prediction,
        config,
        eps=1e-8,
        seed=None,
        filename=None,
    ):
        """
        Initialization of the Metrics class.
        """
        # Store hyperparameters
        self.filename = filename
        self.config = config
        self.seed = seed
        self.eps = eps

        # Store the target solution, mean prediction, and variance prediction
        self.y = target_solution.detach().cpu()
        self.mu = mean_prediction.detach().cpu()
        if var_prediction is None:
            self.var = None
            self.std = None
        else:
            self.var = var_prediction.detach().cpu()
            self.std = torch.sqrt(self.var)

    def _nll(self):
        """
        Compute the Negative Log-Likelihood (NLL) for a Gaussian distribution.
        """
        # Return None if variance is not available
        if self.var is None:
            return None

        return 0.5 * torch.mean(
            torch.log(self.var) + (self.y - self.mu) ** 2 / self.var
        )

    def _ece(self, num_points=100):
        """
        Compute the Expected Calibration Error (ECE).
        """
        # Return None if variance is not available
        if self.var is None:
            return None

        # Compute the CDF values for each prediction
        z = ((self.y - self.mu) / self.std).flatten()
        cdf_vals = 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))

        # Create bins
        ps = torch.linspace(0.0, 1.0, num_points)
        errors = torch.empty_like(ps)

        # For each bin, compute the observed frequency and the absolute error
        for i, p in enumerate(ps):
            observed = (cdf_vals <= p).float().mean()
            errors[i] = torch.abs(observed - p)

        return torch.trapz(errors, ps)

    def _spearson(self):
        """
        Compute the Spearman correlation between the mean squared error and the
        predicted variance.
        """
        # Return None if variance is not available
        if self.var is None:
            return None

        # Compute MSE and variance
        mse = ((self.y - self.mu) ** 2).mean(dim=1).numpy()
        var = self.var.mean(dim=1).numpy()

        return spearmanr(mse, var).correlation

    @staticmethod
    def print_metrics(metrics, title="Metrics"):
        """
        Print the computed metrics in a formatted way.
        """
        # Print a header for the metrics
        print("\n" + "=" * 50)
        print(f"{title:^50}")
        print("=" * 50)

        # Store the maximum key length for formatting
        max_key_len = max(len(k) for k in metrics.keys())

        # Iterate through the metrics
        for key, value in metrics.items():
            if value is None:
                val_str = "N/A"
            elif isinstance(value, float):
                val_str = f"{value: .6e}"
            else:
                val_str = str(value)

            # Print the metric with proper formatting
            print(f"{key.ljust(max_key_len)} : {val_str}")

        # Print a footer for the metrics
        print("=" * 50 + "\n")

    @torch.no_grad()
    def compute(self):
        """
        Compute all the metrics and return them as a dictionary.
        """
        # Compute the error between the target solution and the mean prediction
        error = self.y - self.mu

        # Compute L1 and L2 norms of the error
        l1 = torch.mean(torch.norm(error, p=1, dim=1))
        l2 = torch.mean(torch.norm(error, p=2, dim=1))

        # Compute relative L1 and L2 norms
        rel_l1 = torch.mean(
            torch.norm(error, p=1, dim=1)
            / (torch.norm(self.y, p=1, dim=1) + self.eps)
        )
        rel_l2 = torch.mean(
            torch.norm(error, p=2, dim=1)
            / (torch.norm(self.y, p=2, dim=1) + self.eps)
        )

        # Compute Mean Squared Error (MSE) and Mean Absolute Error (MAE)
        mse = torch.mean(error**2)
        mae = torch.mean(torch.abs(error))

        # Store all metrics in a dictionary
        metrics = {
            "seed": self.seed,
            "nll": self._nll().item() if self.var is not None else None,
            "ece": self._ece().item() if self.var is not None else None,
            "spearson_correlation": self._spearson(),
            "l2": l2.item(),
            "relative_l2": rel_l2.item(),
            "mse": mse.item(),
            "mae": mae.item(),
            "l1": l1.item(),
            "relative_l1": rel_l1.item(),
        }

        # Save metrics to a CSV file
        output_path = os.path.join(
            self.config.output_dir,
            self.config.model_name,
            self.config.name,
        )
        os.makedirs(output_path, exist_ok=True)

        # File path for the CSV file
        file_path = os.path.join(output_path, self.filename)
        file_exists = os.path.isfile(file_path)

        # Write metrics to the CSV file
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(metrics.keys())
            writer.writerow(metrics.values())

        return metrics
