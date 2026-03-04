from src.run import run_training, run_test
from omegaconf import OmegaConf
import hydra
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):

    # Set running directory
    OmegaConf.set_struct(config, False)
    config.experiment.running_path = os.getcwd()
    OmegaConf.set_struct(config, True)

    # run train and test
    if config.experiment.run_training:
        run_training(config.experiment)
    else:
        run_test(config.experiment)


if __name__ == "__main__":
    main()
