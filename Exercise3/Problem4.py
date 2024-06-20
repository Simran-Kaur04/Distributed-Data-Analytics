# Importing Libraries
import os
import numpy as np
from mpi4py import MPI
import ast

# Setting MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
set_files = None
Idf = None
idf_corpus = None

startTime = MPI.Wtime()
# Master Node
if rank == 0:
    file1 = open("Termfrequency.txt", "r")     # stores Term Frequency
    file2 = open("InvTermfrequency.txt", "r")                 # stores Inverse term Frequency
    doc_termfreq = []  # List containing dictionaries having term frequencies
    for doc in file1:
        doc_termfreq.append(ast.literal_eval(doc))

    step = len(doc_termfreq)//size
    set_files = []     # stores chunks containing dictionaries for each worker
    for i in range(size - 1):
        set_files.append(doc_termfreq[i*step: (i + 1)*step])
    set_files.append(doc_termfreq[(size - 1)*step: ])

    idf_corpus = []              
    for tup in file2:
        idf_corpus.append(ast.literal_eval(tup))
    idf_corpus = dict(idf_corpus)             # dictionary containing IDF for entire corpus

Idf = comm.bcast(idf_corpus, root = 0)            # IDF is broadcasted to each worker
worker_chunk = comm.scatter(set_files, root = 0)         # Dictionaries containing term frequencies are scattered among all workers

Tf_Idf = {}
for i, dict in enumerate(worker_chunk):
    token_count = {}                               # TF and IDF are multiplied together to get TFIDF for each token in each document
    for token in dict.keys():
        if token in Idf.keys():
            token_count[token] = dict[token]*Idf[token]
    Tf_Idf['doc' + str(rank*(len(worker_chunk) - 1) + i)] = token_count

tfidf_data = comm.gather(Tf_Idf, root = 0)           

if rank == 0:
    TFIDF = {}           # adding all dictionaries to one big dictionary from all workers
    for d in tfidf_data:
        TFIDF.update(d)
    with open('TfIdf.txt', 'a') as f:     # Saving to a text file where each row contains information of TFIDF for each document
        for item in TFIDF.items():
            f.write(str(item))
            f.write('\n')
    print('Time:', MPI.Wtime()-startTime)


