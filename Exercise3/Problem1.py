# Importing Libraries
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from mpi4py import MPI

# Setting MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
set_files = None

startTime = MPI.Wtime()
# Master node
if rank == 0:
    main = "C:/DDA/20_newsgroups"    # main path of the folder
    directory = os.scandir(main)
    list_files = []      # list stores the path of the subfolders
    for i in directory:
        newpath = os.path.join(i)
        newpath2 = main + '/' + newpath.split("\\")[-1]
        list_files.append(newpath2)

    num_files = len(list_files)
    step = num_files//size

    set_files = []     # stores chunks of paths for each worker
        
    for i in range(size - 1):
        set_files.append(list_files[i*step:(i+1)*step])
    set_files.append(list_files[(size - 1)*step: ])

worker_chunk = comm.scatter(set_files, root = 0)

# cleaning and tokenizing data
clean_data = []
for path in worker_chunk:
    sub_directory = os.scandir(path)
    for j in sub_directory:                 # text files in each folder provided to workers
        files = []
        filepath = os.path.join(j)
        newpath3 = path + '/' + filepath.split("\\")[-1]
        f = open(newpath3, "r")
        file = f.read()
        file = re.sub(r'[^\w\s]', '', file)        # removing punctuations
        file = re.sub(r'[_]+', '', file)          # removing _ as it is part of \w
        file = re.sub(r'[\d]+', '', file)            # removing digits
        file= re.sub(r'\b\w{1}\b', '', file)        # removing words consisting of single characters
        file = re.sub(r'\b\w{15,}\b', '', file)       # removing words of length 15 or more(consists mainly of Urls)
        tokens = nltk.word_tokenize(file)           # tokenizing data
        stop_words = stopwords.words('english')        # list of stopwords
        for word in tokens:
            if word.lower() not in stop_words:        # appending words other than stopwords
                files.append(word.lower())
        clean_data.append(files)

processed_data = comm.gather(clean_data, root = 0)      # gathering cleaned and tokenized data from workers

# Each worker returns list of lists where inner list consists of tokens from each text file

if rank == 0:
    tokenized_data = [item for sublist in processed_data for item in sublist]   # converting list of lists of lists to list of lists
    with open('tokenized_data.txt', 'a') as f:    # saving cleaned data into text file to use in other parts
        for doc in tokenized_data:
            for line in doc:
                f.write(line)
                f.write(' ')             # individual words in text file are seperated using spaces
            f.write('\n')                  # new document starts from new line
    print('Time:', MPI.Wtime()-startTime)








