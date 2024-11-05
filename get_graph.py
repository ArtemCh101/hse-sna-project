import ast
import os
import networkx as nx
import pandas as pd
import datetime


def get_digraph(edgelist_path: str = None, add_attributes: bool = True, attributes_path: str = None) -> nx.DiGraph:
    """
    Create a directed graph from Edgelist.txt

    :param edgelist_path: path to the `Edgelist.txt` file (default is the intended project structure)
    :param add_attributes: True if you want to add node attributes with data from `processed.csv` (by default True)
    :param attributes_path: path to the `processed.csv` file (default is the intended project structure). Ignored if `add_attributes` is False 
    :returns: nx.DiGraph 
    """
    if not edgelist_path:
        g = nx.read_edgelist(os.path.join('data', 'Edgelist.txt'), nodetype=int, create_using=nx.DiGraph)
    else:
        g = nx.read_edgelist(edgelist_path, nodetype=int, create_using=nx.DiGraph)
    if add_attributes:
        if not attributes_path:
            node_properties = pd.read_csv(os.path.join('data', 'processed.csv')).drop(columns=['Unnamed: 0']).set_index('Paper_ID')
        else:
            node_properties = pd.read_csv(attributes_path).drop(columns=['Unnamed: 0']).set_index('Paper_ID')
        node_properties['Date'] = node_properties['Date'].map(lambda s: datetime.datetime.strptime(s, '%Y-%m-%d').date())
        node_properties['Authors'] = node_properties['Authors'].map(ast.literal_eval)
        attributes = node_properties.to_dict(orient='index')
        nx.set_node_attributes(g, attributes)
    return g