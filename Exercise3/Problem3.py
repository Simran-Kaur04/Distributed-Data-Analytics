# Importing Libraries
import math
import os
import numpy as np
from mpi4py import MPI
import ast

# Setting MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

set_files = None

startTime = MPI.Wtime()
# Master node
if rank == 0:
    file = open("Termfrequency.txt", "r") # File containing dictionaries which contains term frequency for each document
    doc_termfreq = []
    for doc in file:
        doc_termfreq.append(ast.literal_eval(doc)) # append all dictionaries as items in a list

    corpus_size = len(doc_termfreq)    # Corpus size(number of documents)
    step = corpus_size//size

    set_files = []                 # stores chunks containing dictionaries for each worker
    for i in range(size - 1):
        set_files.append(doc_termfreq[i*step: (i + 1)*step])
    set_files.append(doc_termfreq[(size - 1)*step: ])

worker_chunk = comm.scatter(set_files, root = 0)

token_count = {}           # Each worker calculates in how many documents it has recieved individual token appears
for dict in worker_chunk:
    for token in dict.keys():
        if token not in token_count.keys():
            token_count[token] = 1
        else:
            token_count[token] += 1

count_data = comm.gather(token_count, root = 0) # gathering dictionaries from each worker containing 
                                                        # number of documents where token appears

if rank == 0:   # The master calculates for the entire corpus
    inv_termfreq = {}
    for dict in count_data:   # contains list of dictionaries
        for key in dict.keys():
            if key not in inv_termfreq.keys():
                inv_termfreq[key] = dict[key]
            else:
                inv_termfreq[key] += dict[key]

    IDF = {}                                  # Calculates IDF
    for k, v in inv_termfreq.items():
        IDF[k] = math.log10(corpus_size/v)
    with open('InvTermfrequency.txt', 'a') as f:         #Saving results to a text file
        for item in IDF.items():                        # Each line of file contains tupple containing token and respective IDF
            f.write(str(item))
            f.write('\n')
    print('Time:', MPI.Wtime()-startTime)
