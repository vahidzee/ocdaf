import os
import uuid

from cdt.utils.R import launch_R_script
import pandas as pd
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))


def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output


def cam_pruning(adj_matrix: np.ndarray, data: np.ndarray, cutoff):
    save_path = os.path.join(_DIR, "score_cam_pruning")

    data_csv_path = np_to_csv(data, save_path)
    dag_csv_path = np_to_csv(adj_matrix, save_path)

    arguments = dict()
    arguments['{PATH_DATA}'] = data_csv_path
    arguments['{PATH_DAG}'] = dag_csv_path
    arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
    arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
    arguments['{CUTOFF}'] = str(cutoff)
    arguments['{VERBOSE}'] = "TRUE"

    def retrieve_result():
        A = pd.read_csv(arguments['{PATH_RESULTS}']).values
        os.remove(arguments['{PATH_RESULTS}'])
        os.remove(arguments['{PATH_DATA}'])
        os.remove(arguments['{PATH_DAG}'])
        return A

    dag = launch_R_script(os.path.join(_DIR, "score_cam_pruning/cam_pruning.R"), arguments,
                          output_function=retrieve_result)
    return dag
