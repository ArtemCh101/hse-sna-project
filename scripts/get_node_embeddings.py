from node2vec import Node2Vec
import os
import numpy as np
import sys
scripts_dir = os.path.join(os.path.dirname(os.path.abspath('')), 'scripts')
if not scripts_dir in sys.path:
    sys.path.append(scripts_dir)
from past_present_train_test_split import prepare_training_data

np.random.seed(10)
G, _, (_, _) = prepare_training_data()

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