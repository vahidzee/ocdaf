import bnlearn
import requests
import tempfile
from tqdm import tqdm
import os
import gzip
import shutil
import typing as th


def get_bnlearn_dag(name, import_configs=None):
    """
    Get the DAG from bnlearn using the given name (or link to the dataset)

    Args:
        name (str): name of the dataset (or link to the dataset)
        import_configs (dict): configs to pass to bnlearn.import_DAG (default is verbose=2)

    Returns:
        bnlearn.BayesianModel: the DAG
    """
    _import_configs = dict(verbose=2)
    _import_configs.update(import_configs if import_configs is not None else {})
    # if name is a link, download it into a temp file
    if name.startswith("http"):
        # download the file, and show the progress
        r = requests.get(name, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        t = tqdm(total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {name}")
        with tempfile.NamedTemporaryFile(delete=False) as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")
        # get the file name
        # if the file is compressed, decompress it into another temp file and show the progress
        if name.endswith(".gz"):
            print(f"Decompressing {name}")
            with gzip.open(f.name, "rb") as f_in:
                with tempfile.NamedTemporaryFile(delete=False) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(f.name)
            f.name = f_out.name

        # import dag from the file
        dag = bnlearn.import_DAG(f.name, **_import_configs)
        # remove the temp file, after importing the dag
        os.remove(f.name)
        return dag

    # load the dag using bnlearn.import_DAG
    dag = bnlearn.import_DAG(name, **_import_configs)
    return dag


def generate_interventions(
    dag,
    num_samples_per_value: int,
    node_list: th.Optional[th.List[str]] = None,
    show_progress: bool = False,
    seed: int = 0,
):
    """
    Generate interventions for the given DAG and node list (default is all nodes)

    Args:
        dag (bnlearn.BayesianModel): the DAG
        num_samples_per_value (int): number of samples per value of the node
            (e.g. 10 means 10 samples for each value of the node so 10*2=20 samples for a binary node)
        node_list (list): list of nodes to intervene on (default is all nodes)
        show_progress (bool): show progress bar for sampling (default: False)
        seed (int): seed for sampling (default: 0)
    """
    # intervene on each node in the dag, or on the nodes in node_list if it's not None
    # for each node, intervene on each value of the node
    # for each intervention, simulate num_samples_per_value samples
    # return a list of interventions, each intervention is a tuple of (node, value, samples)
    interventions = []

    for i, node in enumerate(node_list or dag["model"].nodes()):
        node_interventions = [
            (
                node,
                value,
                dag["model"].simulate(
                    num_samples_per_value, do={node: value}, show_progress=show_progress, seed=seed + i
                ),
            )
            for value in range(dag["model"].get_cpds(node).values.shape[0])
        ]
        interventions.append(node_interventions)
    return interventions
