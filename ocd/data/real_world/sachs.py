import networkx as nx

from ocd.data import OCDDataset
import pandas as pd


class SachsOCDDataset(OCDDataset):
    def __init__(self):
        # load csv file into pandas dataframe
        df = pd.read_csv('sachs_cd3cd28.csv')
        label_mapping = {0: "Raf",
                         1: "Mek",
                         2: "Plcg",
                         3: "PIP2",
                         4: "PIP3",
                         5: "Erk",
                         6: "Akt",
                         7: "PKA",
                         8: "PKC",
                         9: "P38",
                         10: "Jnk"}
        graph = {
            2: [3, 4],
            4: [3],
            7: [1, 5, 6, 10, 9, 0],
            8: [1, 7, 10, 0, 9],
            0: [1],
            1: [5],
            5: [6]
        }
        graph = nx.DiGraph(graph)
        graph = nx.relabel_nodes(graph, label_mapping)

        super().__init__(samples=df, dag=graph, name='Sachs')
