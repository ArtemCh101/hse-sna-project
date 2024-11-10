from node2vec import Node2Vec
import networkx as nx
import os

G = nx.read_edgelist(os.path.join(os.pardir, 'data', 'edgelist.txt'))

n2v = Node2Vec(
    G, 
    dimensions = 64,
    walk_length = 80,
    workers=4,
)

model = n2v.fit(
    window=5,
    min_count=1,
    seed=239
)

model.wv.save_word2vec_format(os.path.join(os.pardir, 'n2c_embs'))
model.save(os.path.join(os.pardir, 'n2c_model'))