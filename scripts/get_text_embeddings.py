from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
import os
import gensim.downloader as api
import sys
scripts_dir = os.path.join(os.path.dirname(os.path.abspath('')), 'scripts')
if not scripts_dir in sys.path:
    sys.path.append(scripts_dir)
from past_present_train_test_split import prepare_training_data

def get_weighted_text_embedding(text, word_vectors, tfidf_weights):
    words = text.split()
    word_embeddings = []
    weights = []
    for word in words:
        if word in word_vectors and word in tfidf_weights:
            word_embeddings.append(word_vectors[word] * tfidf_weights[word])
            weights.append(tfidf_weights[word])

    if not word_embeddings:
        return None

    weighted_embedding = np.average(word_embeddings, axis=0, weights=weights)
    return weighted_embedding

_, df, (_, _) = prepare_training_data()

emb_size = 50
word_vectors = api.load(f"glove-wiki-gigaword-{emb_size}")
texts = df['Abstract'].tolist()

vectorizer = TfidfVectorizer()
vectorizer.fit(texts)
tfidf_matrix = vectorizer.transform(texts)
tfidf_feature_names = vectorizer.get_feature_names_out()


text_embeddings = []
for i, text in tqdm(enumerate(texts), total=len(texts)):
    tfidf_weights = {word: tfidf_matrix[i, idx] for word, idx in zip(
        tfidf_feature_names, range(len(tfidf_feature_names)))}
    embedding = get_weighted_text_embedding(text, word_vectors, tfidf_weights)
    if embedding is None:
        text_embeddings.append(np.zeros_like(text_embeddings[-1]))
    else:
        text_embeddings.append(embedding)

np.save(os.path.join(os.pardir, f'embeddings_{emb_size}.npy'), np.array(text_embeddings))