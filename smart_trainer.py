"""
This is a smart trainer that fills out a lot of the configuration fields for you.
For example, you don't need to fill in the permutation size in both the model 
configuration and the data configuration. This trainer will do it for you.
In addition, it will give you visualization capabilities for the causal discovery. In particular,
if the number of covariates are small and visualizing the permutations on a Birkhoff Polytope is 
feasible, it will do so.

If you run the trainer with the "--discovery" flag, it will find the ordering of the covariates.
In addition to that flag, if you influde the "--inference" flag, after saving the correct ordering in 
a Json file, it gets that correct ordering and runs the model on it using "fixed_permutation" mode. 
This way, we will ensure that all the parameters are put to good use.

Moreover, the epoch_limit for the discovery phase equals (n^2) * (max_epoch) where max_epoch
is the value set in the default configuration it is being passed in. This way, we will ensure
that the model has enough time to discover the ordering.
"""
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch import LightningModule, LightningDataModule
from lightning_toolbox import TrainingModule, DataModule
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from jsonargparse.namespace import Namespace
import typing as th
import dypy as dy
from ruamel import yaml
import sys
import wandb
import json
from jsonargparse import ArgumentParser, ActionConfigFile
import lightning.pytorch as pl
import pprint
import time
import datetime

original_max_epoch: int
checkpoint_path: th.Optional[str] = None


def convert_to_dict(config: th.Union[Namespace, list, dict]) -> th.Union[dict, list]:
    """
    Takes in a namespace or list of namespaces and converts it to a dictionary.
    """
    if isinstance(config, Namespace):
        ret_config = {}
        for key, value in vars(config).items():
            ret_config[key] = convert_to_dict(value)
    elif isinstance(config, list):
        ret_config = [convert_to_dict(item) for item in config]
    elif isinstance(config, dict):
        ret_config = {}
        for key, value in config.items():
            ret_config[key] = convert_to_dict(value)
    else:
        ret_config = config
    return ret_config

def get_callbacks_with_class_path(callbacks_conf, class_path: str):
    indices = []
    for i, callback in enumerate(callbacks_conf):
        if callback["class_path"] == class_path:
            indices.append(i)
    return indices

def handle_birkhoff(new_config, n, graph):
    # Handle Birkhoff Callback, if the number of nodes is less than or equal to 4
    # then edit or add the Birkhoff callback to the callbacks list
    # if not, remove it entirely
    callback_configs = new_config["trainer"]["callbacks"]
    birkhoff_callback_indices = get_callbacks_with_class_path(callback_configs, "ocd.training.callbacks.birkhoff_visualizer.BirkhoffCallback")
    
    if n <= 4 and len(birkhoff_callback_indices) > 0:
        ind_of_interest = birkhoff_callback_indices[0]
        # Change the Birkhoff config to fit the new graph
        # 1. change the permutation size
        callback_configs[ind_of_interest]["init_args"]["permutation_size"] = n
        # 2. change the graph
        callback_configs[ind_of_interest]["init_args"]["causal_graph"] = {
            "class_path": "networkx.classes.digraph.DiGraph",
            "init_args": {"incoming_graph_data": list(graph.edges)},
        }
        birkhoff_callback_indices.pop(0)
    for i, ind in enumerate(birkhoff_callback_indices):
        callback_configs.pop(ind - i)
    new_config["trainer"]["callbacks"] = callback_configs
    
    return new_config

def handle_permutation_saving(new_config, graph, logger_name):
    # Handle Permutation Statistics Callback
    # 1. Change the permutation size
    # 2. Change the graph
    callback_configs = new_config["trainer"]["callbacks"]
    indices = get_callbacks_with_class_path(callback_configs, "ocd.training.callbacks.save_results.SavePermutationResultsCallback")
    if len(indices) > 1:
        raise Exception("There are more than one permutation saving callbacks in the configuration.")
    elif len(indices) == 0:
        raise Exception("There is no permutation saving callback in the configuration.")
    else:
        ind_of_interest = indices[0]
    
    old_save_path = None
    if "save_path" in callback_configs[ind_of_interest]["init_args"]:
        old_save_path = callback_configs[ind_of_interest]["init_args"]["save_path"]
    if old_save_path is None:
        callback_configs[ind_of_interest]["init_args"]["save_path"] = f"experiments/smart/smart-trainer-logs/smart-{logger_name}"
    else:
        callback_configs[ind_of_interest]["init_args"]["save_path"] = os.path.join(old_save_path, f"smart-{logger_name}")
    new_save_path = callback_configs[ind_of_interest]["init_args"]["save_path"]
    if not os.path.exists(new_save_path):
        os.makedirs(new_save_path)
            
    callback_configs[ind_of_interest]["init_args"]["causal_graph"] = {
        "class_path": "networkx.classes.digraph.DiGraph",
        "init_args": {"incoming_graph_data": list(graph.edges)},
    }
    new_config["trainer"]["callbacks"] = callback_configs
    
    return new_config

def handle_base_distributions(new_config):
    distr_name = "torch.distributions.normal.Normal"
    distr_args = {"loc": 0.0, "scale": 1.0}
    if (
        new_config["data"]["init_args"]["dataset_args"] is not None
        and "scm_generator" in new_config["data"]["init_args"]["dataset_args"]
        and new_config["data"]["init_args"]["dataset_args"]["scm_generator"]
        == "ocd.data.synthetic.ParametricSCMGenerator"
    ):
        if new_config["data"]["init_args"]["dataset_args"]["scm_generator_args"]["noise_type"] == "uniform":
            distr_name = "torch.distributions.normal.Normal"
            distr_args = {"loc": 0.0, "scale": 0.33}
        elif new_config["data"]["init_args"]["dataset_args"]["scm_generator_args"]["noise_type"] == "laplace":
            distr_name = "torch.distributions.laplace.Laplace"
            distr_args = {"loc": 0.0, "scale": 1.0}

    if "base_distribution" not in new_config["model"]["init_args"]["model_args"]:
        # If the model args Does not contain the base distribution explicitely then specify it using the values
        # if not, leave as is.
        new_config["model"]["init_args"]["model_args"]["base_distribution"] = distr_name
        new_config["model"]["init_args"]["model_args"]["base_distribution_args"] = distr_args
        
    return new_config

def change_config_for_causal_discovery(old_config, bypass_logger: bool = False):
    """
    This function takes in a base configuration and changes it for the causal discovery phase.
    In this phase, the ordering will be discovered.

    For better interpretability, we will use the Birkhoff polytope to visualize the
    exploration if the number of covariates is less than or equal to 4.

    Finally, we end up with a configuration that is ready to be used by the Lightning Trainer.
    """
    new_config = convert_to_dict(old_config)
    
    # Build the graph and get the graph information
    dataset_args = (
        new_config["data"]["init_args"]["dataset_args"] if "dataset_args" in new_config["data"]["init_args"] else {}
    )
    dataset_args = dataset_args or {}
    construction_args_copy = dataset_args.copy()
    # Disable simulation for runtime acceleration purposes
    if new_config['data']['init_args']['dataset'] == 'ocd.data.SyntheticOCDDataset':
        construction_args_copy['enable_simulate'] = False
        
    torch_dataset = dy.eval(new_config["data"]["init_args"]["dataset"])(**construction_args_copy)
    graph = torch_dataset.dag
    n = graph.number_of_nodes()

    # Handle the logger
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger_name = f"{torch_dataset.name}-{current_timestamp}"
    
    # Change the logger name
    if not bypass_logger:
        new_config["trainer"]["logger"]["init_args"]["name"] = f"discovery-{logger_name}"

    new_config = handle_birkhoff(new_config, n, graph)
    new_config = handle_permutation_saving(new_config, graph, logger_name)

    # Change the model in_features and dimensions
    new_config["model"]["init_args"]["model_args"]["in_features"] = n

    new_config = handle_base_distributions(new_config)

    if bypass_logger:
        return new_config, logger_name

    return Namespace(new_config), logger_name


def change_config_for_causal_inference(old_config, logger_name, correct_ordering):
    new_config = convert_to_dict(old_config)
    new_config["trainer"]["logger"]["init_args"]["name"] = f"inference-{logger_name}"

    # Handle the logger

    # Remove all the unnecessary callbacks:
    # 1. Birkhoff Callback
    # 2. Phase Changer
    # 3. Permutation Statistics Callback
    # 4. Permutation result saver
    indices_to_remove = []
    for i, callback in enumerate(new_config["trainer"]["callbacks"]):
        if callback["class_path"] in [
            "ocd.training.callbacks.birkhoff_visualizer.BirkhoffCallback",
            "ocd.training.callbacks.phase_changer.PhaseChangerCallback",
            "ocd.training.callbacks.save_results.SavePermutationResultsCallback",
            "ocd.training.callbacks.permutation_statistics.PermutationStatisticsCallback",
        ]:
            indices_to_remove.append(i)
    for i, ind in enumerate(indices_to_remove):
        new_config["trainer"]["callbacks"].pop(ind - i)

    # Change the model into fixed ordering
    new_config["model"][
        "class_path"
    ] = "lightning_toolbox.TrainingModule"  # Change the trainer to a standard TrainingModule from lightning_toolbox
    new_config["model"]["init_args"].pop("maximization_specifics")
    new_config["model"]["init_args"].pop("expectation_specifics")
    new_config["model"]["init_args"].pop("grad_clip_val")
    new_config["model"]["init_args"].pop("phases")
    new_config["trainer"]["gradient_clip_val"] = 1.0
    new_config["trainer"]["gradient_clip_algorithm"] = "value"

    new_config["model"]["init_args"]["model_args"]["use_permutation"] = False
    new_config["model"]["init_args"]["model_args"]["ordering"] = correct_ordering
    new_config["model"]["init_args"]["model_args"]["permutation_learner_cls"] = None
    new_config["model"]["init_args"]["model_args"]["permutation_learner_args"] = None

    # Setup optimizers by only cosidering the first optimizer and scheduler
    new_config["model"]["init_args"]["lr"] = new_config["model"]["init_args"]["lr"][0]
    new_config["model"]["init_args"]["optimizer"] = new_config["model"]["init_args"]["optimizer"][0]
    new_config["model"]["init_args"]["optimizer_parameters"] = new_config["model"]["init_args"][
        "optimizer_parameters"
    ][0]
    new_config["model"]["init_args"]["optimizer_is_active"] = None
    new_config["model"]["init_args"]["scheduler"] = new_config["model"]["init_args"]["scheduler"][0]
    new_config["model"]["init_args"]["scheduler_args"] = new_config["model"]["init_args"]["scheduler_args"][0]
    new_config["model"]["init_args"]["scheduler_name"] = "my_inference_scheduler"
    new_config["model"]["init_args"]["scheduler_optimizer"] = new_config["model"]["init_args"]["scheduler_optimizer"][
        0
    ]
    new_config["model"]["init_args"]["scheduler_monitor"] = "loss/val"

    # Add a learning rate monitor callback
    new_config["trainer"]["callbacks"].append(
        {
            "class_path": "lightning.pytorch.callbacks.LearningRateMonitor",
        }
    )

    # Add an EvaluateFlow callback
    new_config["trainer"]["callbacks"].append(
        {
            "class_path": "ocd.training.callbacks.evaluate_flow.EvaluateFlow",
            "init_args": {
                "evaluate_every_n_epochs": 10,
            },
        }
    )

    global original_max_epoch
    new_config["trainer"]["max_epochs"] = original_max_epoch
    return Namespace(new_config), logger_name


def custom_run(conf: th.Union[Namespace, dict], phase: str, log_dir: str):
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")
    sys.argv = sys.argv[:1]
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        # auto_configure_optimizers=False,
        run=False,
        args=conf,
    )
    cli.parser.save(
        conf,
        f"{log_dir}/{phase}-config.yaml",
        skip_none=False,
        overwrite=True,
        multifile=False,
    )
    global checkpoint_path
    cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=checkpoint_path)


def main():
    has_discovery = False
    has_inference = False

    if "--discovery" in sys.argv:
        has_discovery = True
        sys.argv.remove("--discovery")
    if "--inference" in sys.argv:
        has_inference = True
        sys.argv.remove("--inferece")
    if "fit" not in sys.argv:
        raise Exception("Please specify the fit command, the smart trainer only works on fit mode of lightningCLI")
    else:
        sys.argv.remove("fit")
        for i, t in enumerate(sys.argv):
            if t.startswith("--ckpt_path="):
                global checkpoint_path
                checkpoint_path = t.split("=")[1]
                sys.argv.pop(i)
                break

    # parser.parse_object()
    cli = LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        # auto_configure_optimizers=False,
        run=False,
    )
    parser = cli.parser
    wandb.finish()
    base_config = parser.parse_args()

    if has_discovery:
        # Overwrite the base configuration with the causal discovery configuration
        config_for_discovery, logger_name = change_config_for_causal_discovery(base_config)
        saving_callback_ind = get_callbacks_with_class_path(config_for_discovery['trainer']['callbacks'], 
                                                            "ocd.training.callbacks.save_results.SavePermutationResultsCallback")[0]
        
        log_dir = config_for_discovery["trainer"]["callbacks"][saving_callback_ind]["init_args"]["save_path"]

        custom_run(config_for_discovery, phase="causal_discovery", log_dir=log_dir)
        wandb.finish()

        with open(
            f"{log_dir}/final-results.json",
            "r",
        ) as f:
            results = json.load(f)

        correct_permutation = [int(i) for i in results["most_common_permutation"].split("-")]

        if has_inference:
            # Overwrite the base configuration with the causal inference configuration
            config_for_inference, logger_name = change_config_for_causal_inference(
                config_for_discovery, logger_name, correct_permutation
            )
            custom_run(config_for_inference, phase="causal_inference", log_dir=log_dir)
            wandb.finish()
    else:
        raise Exception(
            "Enter the --discovery option in command line to run causal discovery, without that option, the code will not run."
        )


if __name__ == "__main__":
    # delete config.yaml file if it exists
    #
    # This is due to an exception in the LightningCLI
    # that occurs when the config.yaml file already exists
    if os.path.exists("config.yaml"):
        os.remove("config.yaml")

    main()
