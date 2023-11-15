import hydra
import wandb
import os
import yaml

from omegaconf import OmegaConf
from pprint import pprint
from ocd.config import MainConfig
from random_word import RandomWords

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)


def init_run_dir(conf: MainConfig) -> MainConfig:
    # Handle preemption and resume
    run_name = conf.wandb.run_name
    resume = True
    if run_name is None:
        r = RandomWords()
        w1, w2 = r.get_random_word(), r.get_random_word()
        run_name = f"{w1}_{w2}"

    out_dir = os.path.join(conf.out_dir, run_name)

    config_yaml = os.path.join(out_dir, "config.yaml")
    if os.path.exists(config_yaml):
        with open(config_yaml) as fp:
            old_conf = MainConfig(**yaml.load(fp, Loader=yaml.Loader))
        run_id = old_conf.wandb.run_id
    else:
        run_id = wandb.util.generate_id()
        resume = False

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        resume = False

    conf.out_dir = out_dir
    conf.wandb.resume = resume
    conf.wandb.run_id = run_id
    conf.wandb.run_name = run_name
    with open(config_yaml, "w") as fp:
        yaml.dump(conf.model_dump(), fp, default_flow_style=False)

    return conf



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: MainConfig):
    conf = hydra.utils.instantiate(conf)
    conf = MainConfig(**OmegaConf.to_container(conf))

    if conf.test_run:
        pprint(conf.model_dump())
    else:
        conf = init_run_dir(conf)
        wandb.init(
            dir=conf.out_dir,
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=conf.model_dump(),
            name=conf.wandb.run_name,
            id=conf.wandb.run_id,
            resume="allow" if conf.wandb.resume else False,
            # compatible with hydra
            settings=wandb.Settings(start_method="thread"),
        )


if __name__ == "__main__":
    main()