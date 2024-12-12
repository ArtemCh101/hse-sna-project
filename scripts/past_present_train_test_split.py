from functools import cache
import os, sys
import networkx as nx
import pandas as pd
import numpy as np

scripts_dir = os.path.join(os.path.dirname(os.path.abspath('')), 'scripts')
if not scripts_dir in sys.path:
    sys.path.append(scripts_dir)
from get_graph import get_digraph


def train_test_split_indegree(G, df, input_date='2001-01-01',test_frac=0.1):
    '''
    This function takes NetowrkX Graph G, dataframe df,input_date and test_frac as inputs
    
    It splits the vertecies before input_date into train and test part(according to test_frac)
    The function modifies df, adding information about indegree to the vertecies before
    the input_date into 2 additional columns
    
    Returns the modified dataframe and a pair (train, test), containing Paper_ID
    of train and test parts of the Graph before inout_date respectively
    '''
    input_date = pd.to_datetime(input_date)

    nodes_before_date = [n for n, d in G.nodes(data=True) if pd.to_datetime(d['Date']) < input_date]

    nodes_after_date = [n for n, d in G.nodes(data=True) if pd.to_datetime(d['Date']) >= input_date]

    print(f'Papers after date found by split : {len(nodes_before_date)}')

    np.random.seed(10) 
    
    np.random.shuffle(nodes_before_date) # Splitting dates before input_date
    split_index = int((1 - test_frac) * len(nodes_before_date))
    train, test = nodes_before_date[:split_index], nodes_before_date[split_index:]
    
    # Calculate indegree from nodes after input_date for all nodes before input_date
    indegree_before_date = {}
    for node in G.nodes:
        neighbors = set(n for n in G.predecessors(node) if n in nodes_after_date)
        indegree_before_date[node] = len(neighbors)

    # Add  to the dataframe
    df.loc[df['Paper_ID'].isin(indegree_before_date.keys()), 'target_citation_rate'] = df.loc[df['Paper_ID'].isin(indegree_before_date.keys()), 'Paper_ID'].map(indegree_before_date)
    
    df['target_citation_rate'] = df['target_citation_rate'].astype(int)
    
    return (df,(train, test))

@cache
def prepare_training_data(input_date='2001-01-01', test_frac=0.1):
    '''
    Returns subgraph of graph G, located on the default path, which contains only vertecies
    with Date < input_date; and dataframe with target columns
    Also returns the split by train and test for the vertecies before input_date
    '''
    
    G: nx.DiGraph = get_digraph()

    df = pd.read_csv(os.path.join(os.pardir, 'data', 'processed.csv'))
    
    output_df, (train , test) = train_test_split_indegree(G,df,input_date=input_date,test_frac=test_frac)

    output_df = output_df[output_df['Date'] < input_date]
    
    H = G.subgraph([n for n, d in G.nodes(data=True) if pd.to_datetime(d['Date']) < pd.to_datetime(input_date)])
    
    return (H, output_df, (train , test))
