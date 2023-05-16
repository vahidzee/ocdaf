import os, sys
import numpy as np
import dypy
from pathlib import Path
import yaml

sys.path.append("..")

_DIR = os.path.dirname(os.path.abspath(__file__))
_REAL_WORLD_DIR = os.path.join(_DIR, "../experiments/data/real-world/")
from ocd.evaluation import count_SHD, count_SID
from ocd.post_processing.cam_pruning import cam_pruning
from ocd.post_processing.pc_pruning import pc_based_pruning

def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def get_data_config(data_type, data_num):
    if data_type == "syntren":
        paths = Path(_REAL_WORLD_DIR).glob("data-syntren-*.yaml")
    else:
        paths = Path(_REAL_WORLD_DIR).glob("data-sachs.yaml")
    paths = sorted(list(paths))
    data_path = paths[data_num]
    data_config = yaml.load(open(data_path, "r"), Loader=yaml.FullLoader)["init_args"]
    data_name = str(data_path).split("/")[-1].split(".")[0]
    return data_config


def metrics(order: str, data_config):
    order = [int(n) for n in order.split('-')]
    dag = full_DAG(order)
    dataset = dypy.get_value(data_config["dataset"])
    # get the true dag
    dataset = dataset()
    X = dataset.samples
    # dag = pc_based_pruning(X, order)
    # dag = cam_pruning(dag, X)

    true_dag = dataset.dag

    return count_SID(true_dag, dag), count_SHD(true_dag, dag)


config = get_data_config("sachs", 0)

print(metrics("9-0-7-2-4-8-3-6-10-5-1", config))

