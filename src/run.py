import torch
import os

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.utils import to_dense_batch
from lightning.pytorch import seed_everything

from pina.data.data_module import PinaDataModule
from pina.solver import ReducedOrderModelSolver
from pina.problem.zoo import SupervisedProblem
from pina.optim import TorchOptimizer
from pina import Trainer

from .model import MessagePassingNeuralNetwork, PosteriorNetwork
from .bayesian_solver import BayesianReducedOrderModelSolver
from .utils import CustomMSELoss, plot
from .metrics import Metrics
from .data import load_data


def run_training(config):
    """
    Set up and perform training.
    """
    trainer, data_module = _build_runner(config)
    trainer.fit(trainer.solver, datamodule=data_module)

    return config


@torch.no_grad()
def run_test(config):
    """
    Set up and perform testing, including computing metrics and visualizations.
    """
    # If ensemble, store the checkpoint directory and set it to None
    if config.ensemble == True:
        tmp_ckpt = config.ckpt
        config.ckpt = None

    # Get the trainer, the data module, and the solver
    trainer, data_module = _build_runner(config)
    solver = trainer.solver

    # Restore the checkpoint directory for ensemble if it was set to None
    if config.ensemble == True:
        config.ckpt = tmp_ckpt

    # Get test data
    test_data = data_module.test_dataset.get_all_data()["data"]
    params, targets = test_data["input"], test_data["target"]
    target_solution = targets.x

    # Set target node features to None to ensure they are not used
    targets.x = None

    # Move to device
    if config.accelerator == "cuda" and torch.cuda.is_available():
        params = params.cuda()
        targets = targets.cuda()
        target_solution = target_solution.cuda()
        solver = solver.cuda()

    # Bayesian case
    if config.model_name == "Bayes_GCA_ROM":

        # Compute the variational dropout coefficients
        solver.posterior_forward(params, targets)

        # Compute the output of the model for multiple stochastic forward passes
        output = torch.stack(
            [
                solver.model["reduction_network"].decode(
                    z=solver.model["interpolation_network"](x=params),
                    decoding_graph=targets,
                )
                for _ in range(config.mc_steps)
            ],
            dim=0,
        )

        # Compute mean and variance of the predictions
        prediction = output.mean(0)
        var_pred = output.var(0)

    # Ensemble case
    else:
        if config.ensemble == True:

            # Load all checkpoints
            ckpt_files = [
                os.path.join(config.ckpt, f)
                for f in os.listdir(config.ckpt)
                if f.endswith(".ckpt")
            ]

            # Store predictions from each checkpoint in a list
            outputs = []

            # Loop through each checkpoint and compute predictions
            for ckpt in ckpt_files:

                # Load the checkpoint
                solver.load_state_dict(
                    torch.load(ckpt, map_location=solver.device)["state_dict"]
                )

                # Set the model to evaluation mode and compute predictions
                solver.eval()
                with torch.no_grad():
                    z = solver.model["interpolation_network"](x=params)
                    pred = solver.model["reduction_network"].decode(
                        z=z, decoding_graph=targets
                    )
                    outputs.append(pred.unsqueeze(0))

            # Compute mean and variance of the predictions across the ensemble
            output = torch.cat(outputs, dim=0)
            prediction = output.mean(0)
            var_pred = output.var(0)
            config.ckpt = None

        # Deterministic case
        else:
            z = solver.model["interpolation_network"](x=params)
            prediction = solver.model["reduction_network"].decode(
                z=z, decoding_graph=targets
            )
            var_pred = None

    # Move predictions and targets to dense format for metrics and plotting
    prediction, _ = to_dense_batch(prediction, targets.batch)
    target_solution, _ = to_dense_batch(target_solution, targets.batch)
    if var_pred is not None:
        var_pred, _ = to_dense_batch(var_pred, targets.batch)
    pos, _ = to_dense_batch(targets.pos, targets.batch)

    # Extract seed from checkpoint directory name for metrics logging
    seed_from_ckpt = (
        os.path.split(config.ckpt)[-1].split("_")[0]
        if config.ckpt is not None
        else "0000"
    )
    seed_from_ckpt = int(seed_from_ckpt[:4])

    # Metrics file name
    if config.model_name == "Bayes_GCA_ROM":
        filename = "metrics_bayesian.csv"
    elif config.ensemble == True:
        filename = "metrics_ensemble.csv"
    else:
        filename = "metrics.csv"

    # Compute and print metrics
    metrics = Metrics(
        target_solution=target_solution,
        mean_prediction=prediction,
        var_prediction=var_pred,
        config=config,
        seed=seed_from_ckpt,
        filename=filename,
    )
    Metrics.print_metrics(metrics.compute(), f"Metric {config.name.upper()}")

    # Load the triangulation for plotting
    _, _, triang = load_data(f"data/{config.name}.pt")

    # Iterate over the first few samples and create plots
    for i in range(min(len(prediction), config.max_plots)):

        # Create save directory for plots
        save_dir = os.path.join(
            config.output_dir,
            config.model_name,
            config.name,
            "plots" if var_pred is None else "plots_uq",
            f"{seed_from_ckpt:04d}",
        )

        # Plot the results
        plot(
            pred=prediction.cpu(),
            target=target_solution.cpu(),
            var_prediction=var_pred.cpu() if var_pred is not None else None,
            pos=pos.cpu(),
            triang=triang,
            save_dir=save_dir,
            idx=i,
        )


def _build_runner(config):
    """
    Create the trainer and the data module based on the provided configuration.
    """
    # Load data
    params, data, _ = load_data(
        os.path.join(config.running_path, f"data/{config.name}.pt")
    )

    # Define the supervised problem
    problem = SupervisedProblem(params, data)

    # Create the data module
    seed_everything(42)
    data_module = PinaDataModule(
        problem=problem,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Set up the data module for training or testing
    data_module.setup("fit" if config.run_training else "test")

    # Set seed to ensure reproducibility
    seed_everything(config.seed)

    # Create the model
    model = MessagePassingNeuralNetwork(
        parameter_dimension=config.parameter_dimension,
        pde_dimension=config.pde_dimension,
        num_nodes=data[0].num_nodes,
        edge_feature_dim=2,
        latent_dimension=config.latent_dimension,
        n_layers_encoder=config.n_layers_encoder,
        n_layers_decoder=config.n_layers_decoder,
        n_layers_interpolator=config.n_layers_interpolator,
        hidden_dim_encoder=config.hidden_dim_encoder,
        hidden_dim_decoder=config.hidden_dim_decoder,
        hidden_dim_interpolator=config.hidden_dim_interpolator,
        n_message_layers_encoder=config.n_message_layers_encoder,
        n_message_layers_decoder=config.n_message_layers_decoder,
        n_update_layers_encoder=config.n_update_layers_encoder,
        n_update_layers_decoder=config.n_update_layers_decoder,
    )

    # Define the solver kwargs
    solver_kwargs = {
        "problem": problem,
        "reduction_network": model.reduction_network,
        "interpolation_network": model.interpolation_network,
        "use_lt": False,
        "loss": CustomMSELoss(),
        "optimizer": TorchOptimizer(torch.optim.AdamW, lr=config.lr),
    }
    if config.model_name == "GCA_ROM":
        SolverType = ReducedOrderModelSolver
    elif config.model_name == "Bayes_GCA_ROM":
        if config.inject_uq == "decoder":
            reduction_network_regex = [r".*decoder.*"]
            interpolation_network_regex = [r"^(?!.*).*"]
        elif config.inject_uq == "encoder":
            reduction_network_regex = [r".*encoder.*"]
            interpolation_network_regex = [r"^(?!.*).*"]
        elif config.inject_uq == "interpolator":
            reduction_network_regex = [r"^(?!.*).*"]
            interpolation_network_regex = [r".*"]
        elif config.inject_uq == "autoencoder":
            reduction_network_regex = [r".*"]
            interpolation_network_regex = [r"^(?!.*).*"]
        elif config.inject_uq == "all":
            reduction_network_regex = [r".*"]
            interpolation_network_regex = [r".*"]
        else:
            raise RuntimeError
        SolverType = BayesianReducedOrderModelSolver
        solver_kwargs.update(
            {
                "prior_probability": config.prior_probability,
                "gamma": config.gamma,
                "reduction_network_regex": reduction_network_regex,
                "interpolation_network_regex": interpolation_network_regex,
            }
        )
    else:
        raise RuntimeError(
            "invalid model_name arguments, expected one of [GCA_ROM, Bayes_GCA_ROM]"
        )

    # Define the solver
    solver = SolverType(**solver_kwargs)

    # Warmup the solver and add the posterior network if using Bayesian GCA-ROM
    if config.model_name == "Bayes_GCA_ROM":
        solver.warmup(params[0:1], data[0])
        solver.add_posterior(
            PosteriorNetwork(
                parameter_dimension=params.shape[1],
                node_dimension=data[0].posterior_node_feats.shape[1],
                edge_dimension=data[0].edge_attr.shape[1],
                hidden_dim=config.hidden_dim_posterior,
                num_alphas_node=solver.number_bayesian_layers["node"],
                num_alphas_edge=solver.number_bayesian_layers["edge"],
                num_alphas_param=solver.number_bayesian_layers["param"],
                n_layers=config.n_layers_posterior,
            )
        )

    # Load checkpoint if specified
    if config.ckpt is not None:
        solver.load_state_dict(
            torch.load(config.ckpt, map_location="cpu")["state_dict"]
        )

    # Determine the output path for saving checkpoints and logs
    output_pth = os.path.join(config.output_dir, config.model_name, config.name)

    # Determine the checkpoint name based on the seed for reproducibility
    checkpoint_name = f"{config.seed:04d}_checkpoint"

    # Callback for model checkpointing
    callbacks = [
        ModelCheckpoint(
            dirpath=output_pth,
            filename=checkpoint_name,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            verbose=False,
        )
    ]

    # Enable early stopping if specified in the configuration
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=config.patience,
            mode="min",
        )
    )

    # Create the Tensorboard logger
    logger = TensorBoardLogger(
        save_dir=output_pth, name="logs", version=f"{config.seed:04d}"
    )

    # Define the Trainer
    trainer = Trainer(
        solver=solver,
        max_epochs=config.max_epochs,
        batch_size=config.batch_size,
        accelerator=config.accelerator,
        devices=config.devices,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        callbacks=callbacks,
        logger=logger,
        shuffle=True,
        enable_model_summary=True,
        enable_progress_bar=False,
        log_every_n_steps=-1,
        reload_dataloaders_every_n_epochs=0,
    )
    trainer.logging_kwargs["on_step"] = False
    trainer.logging_kwargs["on_epoch"] = True

    return trainer, data_module
