import datasets
from datasets import load_dataset
import torch
import numpy as np
import faiss

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import pickle

import sys

dataset_path = sys.argv[1]
n_cluster = int(sys.argv[2])

train_query_vectors = []
dataset = load_dataset(path=dataset_path+"baidu-ultr_uva-mlm-ctr/baidu-ultr_uva-mlm-ctr.py",
    name="clicks",
    split="train", # ["train", "test"]
    cache_dir=dataset_path+"baidu-ultr_uva-mlm-ctr/parts",
)
for i in range(len(dataset)):
    if sum(dataset[i]['click']) > 0:
        keys = ['bm25', 'bm25_title', 'bm25_abstract', 'tf_idf', 'tf', 'idf', 'ql_jelinek_mercer_short',
                'ql_jelinek_mercer_long', 'ql_dirichlet', 'document_length', 'title_length', 'abstract_length']
        info = dataset[i]
        positions = info['position']
        vectors = []
        for j in range(len(positions)):
            if positions[j] <= 10:
                vector = []
                for k in keys:
                    vector.append(info[k][j])
                vectors.append(np.array(vector))
        # train_vectors.append(dataset[i]['query_document_embedding'])
        # print(vectors)
        qpp_vector = np.concatenate((np.mean(vectors, axis=0), np.std(vectors, axis=0)))
        # print(qpp_vector)

        train_query_vectors.append(qpp_vector)
# print(train_query_vectors)

test_query_vectors = []
test_dataset = load_dataset(path=dataset_path+"baidu-ultr_uva-mlm-ctr/baidu-ultr_uva-mlm-ctr.py",
    name="annotations",
    split="test", # ["train", "test"]
    cache_dir=dataset_path+"baidu-ultr_uva-mlm-ctr/parts",
)
for i in range(len(test_dataset)):
    # test_vectors.append(test_dataset[i]['query_document_embedding'])
    info = test_dataset[i]
    keys = ['bm25', 'bm25_title', 'bm25_abstract', 'tf_idf', 'tf', 'idf', 'ql_jelinek_mercer_short',
            'ql_jelinek_mercer_long', 'ql_dirichlet', 'document_length', 'title_length', 'abstract_length']
    length = len(info['bm25'])

    vector = []
    for k in keys:
        vector.append(np.mean(info[k]))
    for k in keys:
        vector.append(np.std(info[k]))
    test_query_vectors.append(vector)

n_dims = len(list(train_query_vectors[0]))

kmeans = faiss.Kmeans(n_dims, n_cluster, nredo=5, niter=500, verbose=True)

kmeans.train(np.asarray(train_query_vectors))

index = faiss.IndexFlatL2(n_dims)

index.add(kmeans.centroids)
print('centroids:')
print(kmeans.centroids)

with open('centroids_qpp.txt', 'wb') as f:
    pickle.dump(kmeans.centroids, f)

# D, I: matrix: kmeans_ncentroids * kmeans_topk
train_distribution = np.array([0 for _ in range(n_cluster)])
D, I = index.search(np.array(train_query_vectors), 1)
print(len(I))
for i in I:
    train_distribution[int(i)] += 1
train_distribution = train_distribution / sum(train_distribution)
with open('train_distribution_qpp.txt', 'wb') as f:
    pickle.dump(list(train_distribution), f)

test_distribution = np.array([0 for _ in range(n_cluster)])
D, I = index.search(np.array(test_query_vectors), 1)
print(len(I))
for i in I:
    test_distribution[int(i)] += 1
test_distribution = test_distribution / sum(test_distribution)
with open('test_distribution_qpp.txt', 'wb') as f:
    pickle.dump(list(test_distribution), f)

cluster_list = [_ for _ in range(n_cluster)] + [_ for _ in range(n_cluster)]
dataset_list = ['train' for _ in range(n_cluster)] + ['test' for _ in range(n_cluster)]
data = pd.DataFrame({'cluster': cluster_list, 'distribution': np.concatenate((train_distribution, test_distribution)), 'dataset':dataset_list})

# print(data)
sns.barplot(data, x='cluster', y='distribution', hue='dataset')

plt.savefig("tra_cluster_barplot_qpp.pdf", format="pdf")