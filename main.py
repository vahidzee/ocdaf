import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from ocd.config import TrainingConfig
import pprint


@hydra.main(version_base=None, config_path="configurations")
def main(conf):
    conf = hydra.utils.instantiate(conf)
    pprint.pprint(TrainingConfig(**OmegaConf.to_container(conf)).dict())


if __name__ == "__main__":
    main()
