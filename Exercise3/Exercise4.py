# importing Libraries
import numpy as np
np.random.seed(0)
import pandas as pd
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from mpi4py import MPI

# Setting MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initializing centroids and data chunks to None
centroids = None
data_chunks = None
K = 5             # number of centroids

def K_means_local(data, classes):      # function calculates local means for workers
    local_centroids = []
    for k in range(classes.shape[0]):
        c = 0
        summ = 0
        for i, instance in enumerate(data):    # for each data point the membership array gives the cluster
            if i == k:                             # for that instance
                c += 1
                summ += instance                  # summing over all the instances for the same class
        local_centroids.append(summ/c)
    return local_centroids

startTime = MPI.Wtime()                   
if rank == 0:
    categories = ['alt.atheism','talk.religion.misc', 'comp.graphics', 'sci.space','misc.forsale', 'comp.windows.x', 'talk.politics.misc']
    newsgroups = fetch_20newsgroups(subset = 'train', categories = categories)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(newsgroups.data)
    M, N = vectors.shape
    step = M//size

    centroids = np.random.uniform(0, 1, size = (K, N))
    
    data_chunks = []
    for i in range(size - 1):
        data_chunks.append(vectors[i*step:(i+1)*step, :])
    data_chunks.append(vectors[(size - 1)*step:, : ])

centroid = comm.bcast(centroids, root = 0)
worker_chunk = comm.scatter(data_chunks, root = 0)
M_chunk, N_chunk = worker_chunk.shape
K_chunk = centroid.shape[0]

changeInCentroid = True
while changeInCentroid:
    Distance_matrix = np.zeros((M_chunk, K_chunk))
    for i, ele in enumerate(worker_chunk.toarray()):
        for j, center in enumerate(centroid):
            Distance_matrix[i, j] = np.linalg.norm(ele - center)

    membership = np.argmin(Distance_matrix, axis=1).reshape(-1, 1)

    local_means = K_means_local(worker_chunk.toarray(), membership)

    LocalMeans = comm.gather(local_means, root = 0)

    global_mean = []
    if rank == 0:
        for k in range(K):
            local_sum = 0
            for means in LocalMeans:
                local_sum += means[k]
            global_mean.append(local_sum/K)

        total_clusters = 0
        for i in range(K):
            if all(item in centroid[i] for item in global_mean[i]):
                total_clusters += 1

        if total_clusters == K:
            changeInCentroid = False

    else:
        global_mean = None

    centroid = global_mean
    centroid = comm.bcast(centroid)
    changeInCentroid = comm.bcast(changeInCentroid)
    print('In the while loop')

if rank == 0:
    print(global_mean)
    print('Time:', MPI.Wtime()-startTime)







