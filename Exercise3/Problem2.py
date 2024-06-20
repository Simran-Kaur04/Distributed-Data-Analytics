# Importing Libraries
import os
import numpy as np
from mpi4py import MPI

# Setting MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
set_files = None

startTime = MPI.Wtime()
# Master Node
if rank == 0:
    file = open("tokenized_data.txt", "r") # This file contains words of individual documents in each line seperated by spaces
    word_corpus = []
    for doc in file:
        word_doc = doc.split(' ')[:-1]      
        word_corpus.append(word_doc)      # All the words are appended in a list for each document

    num_files = len(word_corpus)
    step = num_files//size

    set_files = []       # stores chunks containing words for each worker
    for i in range(size - 1):
        set_files.append(word_corpus[i*step: (i + 1)*step])
    set_files.append(word_corpus[(size - 1)*step: ])

worker_chunk = comm.scatter(set_files, root = 0)

term_frequency = []
for docs in worker_chunk:      # for each document in a worker chunk the term frequency of words in that document
    len_doc = len(docs)                           # are calculated
    token_freq = {}
    for word in docs:
        if word not in token_freq.keys():
            token_freq[word] = 1/len_doc
        else:
            token_freq[word] += 1/len_doc
    term_frequency.append(token_freq)

frequency_data = comm.gather(term_frequency, root = 0) # gathering term frequency data from workers

if rank == 0:
    term_doc = [item for sublist in frequency_data for item in sublist]
    with open('Termfrequency.txt', 'a') as f:
        for item in term_doc:
            f.write(str(item))
            f.write('\n')
    print('Time:', MPI.Wtime()-startTime)
