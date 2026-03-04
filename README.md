# Bayesian GCA-ROM

Official repository of the paper *Towards Uncertainty Quantification in Data-Driven Reduced-Order Models via Bayesian Graph Neural Networks*, ICLR 2026 AI&PDE Workshop [[paper](https://openreview.net/forum?id=6JVy8ZbuHf)].

<figure>
  <img src="img/poisson.gif" width="800">
  <figcaption>
  Poisson simulation with varying obstacle position. From left to right: ground-truth solution, predicted mean, squared error, and predictive variance.
  </figcaption>
</figure>

## Description

The proposed method combines GNN–based reduced-order modeling with efficient Bayesian inference via Variational Adaptive Dropout, enabling principled uncertainty quantification for surrogate models of parametric partial differential equations (PDEs). By extending the [GCA-ROM](https://www.sciencedirect.com/science/article/pii/S0021999124000111) with Bayesian message passing, the framework captures epistemic uncertainty while preserving the geometric structure of the underlying discretization.


The resulting model produces calibrated predictive uncertainty on top of accurate predictions while maintaining the computational efficiency required for fast surrogate inference across varying geometries and parametric regimes.

## Installation and Setup

Clone the repository, create a Conda environment, and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/GiovanniCanali/bayesian_gca_rom.git
cd bayesian_gca_rom

# Create and activate the conda environment
conda create --name bayesian_gca_rom python=3.12 -y
conda activate bayesian_gca_rom

# Install the requirements
python -m pip install -r requirements.txt
```

## Running Experiments

### Training

To run the training procedure, execute the following command and select the appropriate options:

```bash
python main.py experiment=<airfoil/poisson> experiment.model_name=<GCA_ROM/Bayes_GCA_ROM> experiment.run_training=True experiment.seed=<seed>
```

- `GCA_ROM` corresponds to the deterministic baseline.
- `Bayes_GCA_ROM` corresponds to the Bayesian model.

Logs and checkpoints are stored in:
```bash
results/<model_name>/<experiment>
```

Additional training settings can be modified through the configuration files in the `configs/` directory, or overridden directly from the command line when launching an experiment.

### Evaluation

To compute metrics and generate plots from a trained model, run:

```bash
python main.py experiment=<airfoil/poisson> experiment.model_name=<GCA_ROM/Bayes_GCA_ROM> experiment.run_training=False experiment.ckpt=<path/to/checkpoint.ckpt>
```

To compute metrics using an ensemble of models, run:

```bash
python main.py experiment=<airfoil/poisson> experiment.model_name=GCA_ROM experiment.run_training=False experiment.ensemble=True experiment.ckpt=<path/to/checkpoints/>
```

Note:
- For a single model, `experiment.ckpt` must point to a checkpoint file.
- For an ensemble, it must point to a directory containing one or more checkpoint files.

> The ensemble option is intended to be used only with the deterministic `GCA_ROM` model.

## Citation
If you find this work useful, please cite:
```
@inproceedings{
    canali2026towards,
    title={Towards Uncertainty Quantification in Data-Driven Reduced Order Models via Bayesian Graph Neural Networks},
    author={Giovanni Canali and Filippo Olivo and Dario Coscia and Nicola Demo and Gianluigi Rozza},
    booktitle={AI{\&}PDE: ICLR 2026 Workshop on AI and Partial Differential Equations},
    year={2026},
    url={https://openreview.net/forum?id=6JVy8ZbuHf}
}
```
