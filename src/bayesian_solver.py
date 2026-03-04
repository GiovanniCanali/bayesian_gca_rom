from pina.solver import ReducedOrderModelSolver
import torch.nn.functional as F
import math, re
import torch


class KL:
    """
    Kullback-Leibler divergence for the Bayesian layers.
    """
    def __init__(self, p=0.2, beta=0.1):
        """
        Initialization of the KL divergence.
        """
        # Validate inputs
        if p < 0 or p > 1:
            raise ValueError("p must be in [0, 1].")
        if beta <= 0:
            raise ValueError("beta must be positive.")
        
        # Precompute constants for KL divergence calculation
        self.prior = p / (1.0 - p)
        self.logit = math.log(p / (1.0 - p))
        self.beta = beta

    def __call__(self, alphas, edge_index=None):
        """
        Compute the KL divergence for the given alpha values.
        """
        # If edge_index is provided, use alphas from edge source nodes
        if edge_index is not None:
            alphas = alphas[edge_index[0]]

        # Store log of alphas
        log_alpha = torch.log(alphas)
    
        # KL divergence per feature dimension, per element (node or edge)
        kl = 0.5 * (((alphas + 1.0) / self.prior) + self.logit - log_alpha - 1)

        return self.beta * kl.mean()


class LinearBayesian(torch.nn.Linear):
    """
    Bayesian linear layer that computes the mean and variance of the output
    based on the input and the alpha values.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
    ):
        """
        Initialization of the LinearBayesian layer.
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        self.tol = 1e-18

        # If bias is None, create a non-trainable bias parameter
        if self.bias is None:
            self.bias = torch.nn.Parameter(
                torch.tensor([0], dtype=self.weight.dtype), requires_grad=False
            )
        
    def forward(self, input, alpha):
        """
        Forward pass through the LinearBayesian layer.
        """
        mean = F.linear(input, self.weight, self.bias)
        var = alpha * F.linear(input.pow(2), self.weight.pow(2))
        if self.training:
            return mean + torch.sqrt(var + self.tol) * torch.randn_like(mean)
        return mean


class BayesianReducedOrderModelSolver(ReducedOrderModelSolver):
    """
    The bayesian counterpart of the GCA-ROM solver.
    """

    def __init__(
        self,
        problem,
        reduction_network,
        interpolation_network,
        prior_probability=0.2,
        gamma=0.1,
        reduction_network_regex=[r".*"],
        interpolation_network_regex=[r".*"],
        loss=None,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=True,
    ):
        """
        Initialization of the BayesianReducedOrderModelSolver.
        """
        # Save regex
        self.reduction_network_regex = reduction_network_regex
        self.interpolation_network_regex = interpolation_network_regex

        # Save the kl loss
        self.kl_loss = KL(p=prior_probability, beta=gamma)

        # Set warmup and posterior added to False
        self._warmup_done = False
        self._posterior_added = False
        self._alpha_counters = None

        # Make the posterior network
        super().__init__(
            problem,
            reduction_network,
            interpolation_network,
            loss,
            optimizer,
            scheduler,
            weighting,
            use_lt,
        )

    def on_fit_start(self):
        """
        Called at the start of the fitting process. Checks if warmup is done and
        posterior is added before proceeding.
        """
        # Check if warmup is done
        if not self._warmup_done:
            raise RuntimeError("warmup must be done, use warmup method.")

        # Check if posterior is added
        if not self._posterior_added:
            raise RuntimeError(
                "posterior must be added, use add_posterior method."
            )

        return super().on_fit_start()

    def add_posterior(self, posterior):
        """
        Add the posterior network to the model and reconfigure the optimizers.
        """
        # Add posterior
        self.model["posterior"] = posterior
        self._posterior_added = True

        # Reconfigure optimizer
        self.configure_optimizers()

    def optimization_cycle(self, batch):
        """
        Compute the loss for the current batch, including both the standard loss
        and the KL divergence for the Bayesian layers.
        """
        condition_loss = {}
        for condition_name, points in batch:

            # Compute posterior
            posterior_output = self.posterior_forward(
                points["input"], points["target"]
            )

            # Compute standard loss
            condition_loss[condition_name] = self.loss_data(
                input=points["input"], target=points["target"]
            )

            # Compute KL divergence
            if self.training:

                kl_node = self.kl_loss(posterior_output["alpha_node"])
                kl_edge = self.kl_loss(
                    posterior_output["alpha_edge"],
                    edge_index=points["target"].edge_index,
                )
                kl_param = self.kl_loss(posterior_output["alpha_param"])

                # Add KL divergence to the condition loss
                condition_loss[condition_name + "_kl"] = (
                    kl_node + kl_edge + kl_param
                )

        return condition_loss

    def posterior_forward(self, params, batch):
        """
        Perform a forward pass through the posterior network and scatter the
        alpha values.
        """
        # Compute posterior and scatter alphas
        posterior_output = self.model["posterior"](params, batch)
        self.scatter_alphas(**posterior_output)

        return posterior_output

    def scatter_alphas(self, alpha_node, alpha_edge, alpha_param):
        """
        Scatter the alpha values to the respective domains.
        """
        self._alphas = {
            "node": alpha_node,
            "edge": alpha_edge,
            "param": alpha_param,
        }

    def warmup(self, params, batch):
        """
        Warmup the model by performing a forward pass to determine the domain of
        each linear layer and then injecting the Bayesian layers based on the
        provided regex patterns.
        """
        # attach hooks to determine the domain of each linear layer
        h = self._domain_hook(batch.num_nodes, batch.num_edges, params.shape[0])
        handles = [
            m.register_forward_pre_hook(h)
            for m in self.model.modules()
            if isinstance(m, torch.nn.Linear)
        ]

        # Trigger the hooks
        self.model["reduction_network"].encode(batch)
        z = self.model["interpolation_network"](params)
        self.model["reduction_network"].decode(z, batch)

        # Free the \hooks
        for handle in handles:
            handle.remove()

        # Activate bayesian layers
        inject_bayesian_layers(
            self.model["interpolation_network"],
            self.interpolation_network_regex,
        )
        inject_bayesian_layers(
            self.model["reduction_network"], self.reduction_network_regex
        )

        # Attach hooks to scatter alpha values during the forward pass
        self._attach_alpha_hooks()

        # Set warmup as finished
        self._warmup_done = True
        if not all(value > 0 for value in self.number_bayesian_layers.values()):
            raise RuntimeError(
                "No valid bayesian layers matched the given regex pattern."
            )

    def _domain_hook(self, num_nodes, num_edges, num_params):
        """
        Create a hook function to determine the domain of each linear layer
        """

        def hook(module, input):
            """
            Determine the domain of the linear layer based on the input size.
            """
            size = input[0].size(0)
            if size == num_nodes:
                module._domain = "node"
            elif size == num_edges:
                module._domain = "edge"
            elif size == num_params:
                module._domain = "param"
            else:
                module._domain = "unknown"

        return hook

    def _attach_alpha_hooks(self):
        """
        Attach hooks to scatter alpha values during the forward pass.
        """
        # Initialize alpha counters for each domain
        alpha_counters = {"node": 0, "edge": 0, "param": 0}

        # Iterate through all modules and attach hooks to Bayesian layers
        for module in self.model.modules():
            for child in module.children():
                if isinstance(child, LinearBayesian):
                    domain = getattr(child, "_domain", "unknown")
                    if domain in alpha_counters:
                        child.alpha_index = alpha_counters[domain]
                        hook_fn = self._alpha_hook(domain)
                        child.register_forward_pre_hook(hook_fn)
                        alpha_counters[domain] += 1

        # Save the alpha counters
        self._alpha_counters = alpha_counters

    def _alpha_hook(self, domain):
        """
        Create a hook function to scatter alpha values during the forward pass.
        """

        def hook(module, input):
            """
            Scatter alpha values during the forward pass.
            """
            alpha = self._alphas[domain][..., module.alpha_index]
            return input[0], self._broadcast_alpha(alpha, input[0].shape)

        return hook

    def _broadcast_alpha(self, alpha, shape_input):
        """
        Broadcast alpha values to match the input shape.
        """
        diff = len(shape_input) - alpha.ndim
        if diff > 0:
            alpha = alpha.view(*alpha.shape, *([1] * diff))
        return alpha

    @property
    def number_bayesian_layers(self):
        """
        Get the number of Bayesian layers for each domain.
        """
        return self._alpha_counters


def inject_bayesian_layers(model, regex_pattern):
    """
    Inject Bayesian layers into the model based on the provided regex pattern.
    """

    def _get_full_path(module, parent_path=""):
        """
        Get the full path of all submodules in the model.
        """
        for name, child in module.named_children():
            full_path = f"{parent_path}.{name}" if parent_path else name
            yield full_path, module, name, child
            yield from _get_full_path(child, full_path)

    # Compile the regex pattern
    pattern = re.compile("|".join(regex_pattern))

    # Iterate through all submodules and inject Bayesian layers where applicable
    for full_path, parent, name, child in _get_full_path(model):

        # Skip quickly if no match
        if not pattern.search(full_path):
            continue

        # Skip is no eligible
        if not isinstance(child, torch.nn.Linear):
            continue

        # Inject Bayesian layer
        new_layer = LinearBayesian(
            in_features=child.in_features,
            out_features=child.out_features,
            bias=hasattr(child, "bias"),
        )

        # Set the domain attribute if it exists in the original layer
        new_layer._domain = getattr(child, "_domain", "unknown")

        # Copy pretrained weights and bias
        if hasattr(child, "weight") and child.weight is not None:
            new_layer.weight.data.copy_(child.weight.data)
        if hasattr(child, "bias"):
            new_layer.bias.data.copy_(child.bias.data)

        # Replace the module
        setattr(parent, name, new_layer)
