import hydra
import wandb

from omegaconf import OmegaConf
from pprint import pprint
from ocd.config import MainConfig

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: MainConfig):
    conf = hydra.utils.instantiate(conf)
    conf = MainConfig(**OmegaConf.to_container(conf))

    if conf.test_run:
        pprint(conf.dict())
    # else:
    #     wandb.init(
    #         dir=conf.out_dir,
    #         project=conf.wandb.project,
    #         entity=conf.wandb.entity,
    #         config=conf.dict(),
    #         name=conf.wandb.run_name,
    #         id=conf.wandb.run_id,
    #         resume="allow" if conf.wandb.resume else False,
    #         # compatible with hydra
    #         settings=wandb.Settings(start_method="thread"),
    #     )


if __name__ == "__main__":
    main()