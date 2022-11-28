import bnlearn
import requests
import tempfile
from tqdm import tqdm
import os
import gzip
import shutil

def get_bnlearn_dag(name, import_configs=None):
    import_configs = import_configs if import_configs is not None else {}
    # if name is a link, download it into a temp file
    if name.startswith('http'):
        
        # download the file, and show the progress
        r = requests.get(name, stream=True)
        total_size = int(r.headers.get('content-length', 0));
        block_size = 1024 #1 Kibibyte
        t=tqdm(
            total=total_size, unit='iB', 
            unit_scale=True, 
            desc=f'Downloading {name}'
        )
        with tempfile.NamedTemporaryFile(delete=False) as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")
        # get the file name
        # if the file is compressed, decompress it into another temp file and show the progress
        if name.endswith('.gz'):
            print(f'Decompressing {name}')
            with gzip.open(f.name, 'rb') as f_in:
                with tempfile.NamedTemporaryFile(delete=False) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(f.name)
            f.name = f_out.name
            

        # import dag from the file
        dag = bnlearn.import_DAG(f.name)
        # remove the temp file, after importing the dag
        os.remove(f.name)
        return dag
        
    # load the dag using bnlearn.import_DAG
    dag = bnlearn.import_DAG(name)
    return dag

def generate_interventions(dag, num_samples_per_value, node_list=None, show_progress=False):
    
    # intervene on each node in the dag, or on the nodes in node_list if it's not None
    # for each node, intervene on each value of the node
    # for each intervention, simulate num_samples_per_value samples
    # return a list of interventions, each intervention is a tuple of (node, value, samples)
    interventions = []

    for node in node_list or dag['model'].nodes():
        node_interventions = [
            (node, value, dag["model"].simulate(num_samples_per_value, do={node: value}, show_progress=show_progress))
            for value in range(dag["model"].get_cpds(node).values.shape[0])
        ]
        interventions.append(node_interventions)
    return interventions

