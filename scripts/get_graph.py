import ast
import os
import networkx as nx
import pandas as pd
import datetime


def get_digraph(edgelist_path: str = None,
                add_attributes: bool = True,
                attributes_path: str = None
                ) -> nx.DiGraph:
    """
    Create a directed graph from the edgelist.

    Parameters
    ----------
    edgelist_path : str
        Path to the `edgelist.txt` file (default is the intended project
        structure).
    add_attributes : bool, default=True
        True if you want to add node attributes with data from `processed.csv`.
    attributes_path: str
        Path to the `processed.csv` file (default is the intended project
        structure). Ignored if `add_attributes` is False.

    Returns
    -------
    G : nx.DiGraph
    """
    data_path = 'data'
    # if we run not from project root folder but from one of these immediate
    # sub-directories, it will still be fine
    if os.path.basename(os.getcwd()) in ['scripts', 'paper', 'notebooks',
                                         'models', 'data']:
        data_path = os.path.join(os.pardir, data_path)

    if not edgelist_path:
        edgelist_path = os.path.join(data_path, 'edgelist.txt')

    g = nx.read_edgelist(edgelist_path,
                         nodetype=int,
                         create_using=nx.DiGraph)
    if add_attributes:
        if not attributes_path:
            attributes_path = os.path.join(data_path, 'processed.csv')

        node_properties = pd.read_csv(attributes_path).set_index('Paper_ID')

        node_properties['Date'] = node_properties['Date'].map(
            lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date())
        node_properties['Authors'] = node_properties['Authors'].map(
            ast.literal_eval)

        attributes = node_properties.to_dict(orient='index')
        nx.set_node_attributes(g, attributes)
    return g


if __name__ == '__main__':
    get_digraph()
