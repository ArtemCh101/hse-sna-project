import ast
import os, sys
import networkx as nx
import pandas as pd
import datetime
import numpy as np

scripts_dir = os.path.join(os.path.dirname(os.path.abspath('')), 'scripts')
if not scripts_dir in sys.path:
    sys.path.append(scripts_dir)
from get_graph import get_digraph


def train_test_split_indegree(G, df, input_date='2001-01-01',test_frac=0.1):
    '''
    This function takes NetowrkX Graph G, dataframe df,input_date and test_frac as inputs
    
    It splits the vertecies after input_date into train and test part(according to test_frac)
    The function modifies df, adding information about indegree to the vertecies before
    the input_date into 2 additional columns
    
    Returns the modified dataframe
    '''
    input_date = pd.to_datetime(input_date)

    nodes_after_date = [n for n, d in G.nodes(data=True) if pd.to_datetime(d['Date']) >= input_date]

    print(f'Papers after date found by split : {len(nodes_after_date)}')

    np.random.shuffle(nodes_after_date)
    split_index = int((1 - test_frac) * len(nodes_after_date))
    train, test = nodes_after_date[:split_index], nodes_after_date[split_index:]
    
    # Calculate indegree from nodes after input_date for all nodes before input_date
    indegree_before_date_train = {}
    indegree_before_date_test = {}
    for node in G.nodes:
        if pd.to_datetime(G.nodes[node]['Date']) <= input_date:
            train_neighbors = set(n for n in G.predecessors(node) if n in train)
            test_neighbors = set(n for n in G.predecessors(node) if n in test)
            indegree_before_date_train[node] = len(train_neighbors)
            indegree_before_date_test[node] = len(test_neighbors)

    # Add  to the dataframe
    df.loc[df['Paper_ID'].isin(indegree_before_date_train.keys()), 'indegree_train'] = df.loc[df['Paper_ID'].isin(indegree_before_date_train.keys()), 'Paper_ID'].map(indegree_before_date_train)
    df.loc[df['Paper_ID'].isin(indegree_before_date_test.keys()), 'indegree_test'] = df.loc[df['Paper_ID'].isin(indegree_before_date_test.keys()), 'Paper_ID'].map(indegree_before_date_test)


    return df

def prepare_training_data(input_date='2001-01-01', test_frac=0.1):
    '''
    Returns subgraph of graph G, placed on the default path, which contains only vertecies
    with Date < input_date; and dataframe with target columns split by train and test
    only for the desired vertecies
    '''
    
    G: nx.DiGraph = get_digraph()

    df = pd.read_csv(os.path.join(os.pardir, 'data', 'processed.csv'))
    
    output_df = train_test_split_indegree(G,df,input_date=input_date,test_frac=test_frac)

    output_df = output_df[output_df['Date'] < input_date]
    
    H = G.subgraph([n for n, d in G.nodes(data=True) if pd.to_datetime(d['Date']) < pd.to_datetime(input_date)])
    
    return (H, output_df)