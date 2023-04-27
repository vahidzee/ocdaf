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

original_max_epoch: int


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


def change_config_for_causal_discovery(old_config):
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
    torch_dataset = dy.eval(new_config["data"]["init_args"]["dataset"])(**dataset_args)
    graph = torch_dataset.dag
    n = graph.number_of_nodes()

    # Handle the logger
    new_config["trainer"]["logger"]["init_args"]["name"]
    logger_name = torch_dataset.name
    mex = 0
    while os.path.exists(f"experiments/smart-trainer-logs/{logger_name}-{mex}"):
        mex += 1
    logger_name = f"{logger_name}-{mex}"
    # create the directory
    os.makedirs(f"experiments/smart-trainer-logs/{logger_name}")
    new_config["trainer"]["logger"]["init_args"]["name"] = f"discovery-{logger_name}"

    # Handle Birkhoff Callback, if the number of nodes is less than or equal to 4
    # then edit or add the Birkhoff callback to the callbacks list
    # if not, remove it entirely
    callback_configs = new_config["trainer"]["callbacks"]
    if n <= 4:
        # Find the Birkhoff config if it exists or add it to the callbacks if it does not
        birkhoff_config = None
        ind_of_interest = None
        for i, callback_config in enumerate(callback_configs):
            if callback_config["class_path"] == "ocd.training.callbacks.birkhoff_visualizer.BirkhoffCallback":
                birkhoff_config = callback_config
                ind_of_interest = i
                break
        if birkhoff_config is not None:
            # Change the Birkhoff config to fit the new graph
            # 1. change the permutation size
            callback_configs[ind_of_interest]["init_args"]["permutation_size"] = n
            # 2. change the graph
            callback_configs[ind_of_interest]["init_args"]["causal_graph"] = {
                "class_path": "networkx.classes.digraph.DiGraph",
                "init_args": {"incoming_graph_data": list(graph.edges)},
            }
    else:
        # Remove the Birkhoff visualizer callback
        indices = []
        for i, callback_config in enumerate(callback_configs):
            if callback_config["class_path"] == "ocd.training.callbacks.birkhoff_visualizer.BirkhoffCallback":
                indices.append(i)
        for i, ind in enumerate(indices):
            callback_configs.pop(ind - i)
    new_config["trainer"]["callbacks"] = callback_configs

    # Handle Permutation Statistics Callback
    # 1. Change the permutation size
    # 2. Change the graph
    callback_configs = new_config["trainer"]["callbacks"]
    perm_save_callback = None
    ind_of_interest = None
    for i, permutation_statistics_callback in enumerate(callback_configs):
        if (
            permutation_statistics_callback["class_path"]
            == "ocd.training.callbacks.save_results.SavePermutationResultsCallback"
        ):
            perm_save_callback = permutation_statistics_callback
            ind_of_interest = i
            break
    if perm_save_callback is None:
        perm_save_callback = {}
        callback_configs.append(perm_save_callback)
        ind_of_interest = len(callback_configs) - 1

    callback_configs[ind_of_interest]["init_args"]["save_path"] = f"experiments/smart-trainer-logs/{logger_name}"
    callback_configs[ind_of_interest]["init_args"]["causal_graph"] = {
        "class_path": "networkx.classes.digraph.DiGraph",
        "init_args": {"incoming_graph_data": list(graph.edges)},
    }
    new_config["trainer"]["callbacks"] = callback_configs

    # Set the max_epoch
    global original_max_epoch
    original_max_epoch = new_config["trainer"]["max_epochs"]
    new_config["trainer"]["max_epochs"] = min(100000, original_max_epoch * n**2)

    # Change the model in_features and dimensions
    new_config["model"]["init_args"]["model_args"]["in_features"] = n
    # TODO: maybe change the hidden layer architecture as well?
    distr_name = "torch.distributions.normal.Normal"
    distr_args = {"loc": 0.0, "scale": 1.0}
    if (
        new_config["data"]["init_args"]["dataset_args"] is not None
        and "scm_generator" in new_config["data"]["init_args"]["dataset_args"]
        and new_config["data"]["init_args"]["dataset_args"]["scm_generator"]
        == "ocd.data.synthetic.LinearNonGaussianSCMGenerator"
    ):
        if new_config["data"]["init_args"]["dataset_args"]["scm_generator_args"]["noise_type"] == "uniform":
            distr_name = "torch.distributions.normal.Normal"
            distr_args = {"loc": 0.0, "scale": 0.33}
        elif new_config["data"]["init_args"]["dataset_args"]["scm_generator_args"]["noise_type"] == "laplace":
            distr_name = "torch.distributions.laplace.Laplace"
            distr_args = {"loc": 0.0, "scale": 1.0}

    new_config["model"]["init_args"]["model_args"]["base_distribution"] = distr_name
    new_config["model"]["init_args"]["model_args"]["base_distribution_args"] = distr_args

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
    cli.trainer.fit(cli.model, cli.datamodule)


def main():
    has_discovery = False
    has_inference = False

    if "--discovery" in sys.argv:
        has_discovery = True
        sys.argv.remove("--discovery")
    if "--inference" in sys.argv:
        has_inference = True
        sys.argv.remove("--inferece")

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
        log_dir = f"experiments/smart-trainer-logs/{logger_name}"

        custom_run(config_for_discovery, phase="causal_discovery", log_dir=log_dir)
        wandb.finish()

        with open(f"experiments/smart-trainer-logs/{logger_name}/final-results.json", "r") as f:
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
