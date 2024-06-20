from numpy import int64
import pandas as pd
import os
import numpy as np
np.printoptions(threshold=10)
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
data_chunks = None
TestData = None

epochs =  10
learning_rate = 10**-(12)


def RMSE(data, beta, target):
    prediction = np.dot(data, beta)
    return np.sqrt((np.sum((prediction - target.reshape(-1,1))**2))/data.shape[0])

def grad(instnce, beta, target):
    return ((2*(np.dot(instnce, beta) - target))*instnce).reshape(-1,1)

start_time = MPI.Wtime()
if rank == 0:
    main = 'C:/Users/simra/DDALab/Exercise4/dataset'
    directory = os.scandir(main)
    files = []
    for i in directory:
        array = []
        newpath = os.path.join(i)
        newpath2 = main + '/' + newpath.split("\\")[-1]
        file = open(newpath2, 'r')
        f = file.read()
        num_lines = sum(1 for line in f.split('\n'))
        np_array = np.zeros((num_lines, 482))
        target_vector = np.zeros((num_lines, 1))
        for idx, line in enumerate(f.split('\n')[:-1]):
            values = {}
            target_vector[idx, 0] = float(line.split(' ')[0])
            array = line.split(' ')[1:-1]
            for k, v in [val.split(':') for val in array]:
                values[int(k)] = int(v)
            for j in range(np_array.shape[1]):
                if j in values.keys():
                    np_array[idx, j] = values[j]
        indx = np.where(~np_array.any(axis=0))[0]
        clean_array = np.delete(np_array, indx, axis=1)
        files.append(np.hstack((np_array, target_vector)))
    Data = np.vstack(files)
    bias_column = np.ones(shape=(Data.shape[0],1))
    Data_biasC = np.append(bias_column, Data ,axis=1)
    TrainData = Data_biasC[: int(0.7*Data_biasC.shape[0]), :]
    TestData = Data_biasC[int(0.7*Data_biasC.shape[0]): , :]
    M, N = TrainData.shape
    step = M//size

    data_chunks = []                  # splitting data for distribution among workers
    for i in range(size - 1):
        data_chunks.append(TrainData[i*step:(i+1)*step, :])
    data_chunks.append(TrainData[(size - 1)*step:, : ])


test = comm.bcast(TestData, root = 0)
worker_chunk = comm.scatter(data_chunks, root = 0)
M_chunk, N_chunk = worker_chunk.shape
old_params = np.zeros((N_chunk-1, 1))
new_params = np.zeros((N_chunk-1, 1))
RmseTrain = []
RmseTest = []

for _ in range(epochs):
    np.random.shuffle(worker_chunk)
    for i in range(M_chunk):
        new_params = old_params - learning_rate*grad(worker_chunk[i, :-1], old_params, worker_chunk[i, -1])
        old_params = new_params.copy()
    RmseTrain.append(RMSE(worker_chunk[:, :-1], new_params, worker_chunk[:, -1]))
    RmseTest.append(RMSE(test[:, :-1], new_params, test[:, -1]))

local_params = comm.gather(new_params, root = 0)
local_errorTrain = comm.gather(RmseTrain, root = 0)
local_errorTest = comm.gather(RmseTest, root = 0)

if rank == 0:
    weights = np.hstack(local_params)
    global_params = np.mean(weights, axis=1)
    errorTrain = np.vstack(local_errorTrain)
    errorTest = np.vstack(local_errorTest)
    global_RmseTrain = np.mean(errorTrain, axis = 0)
    global_RmseTest = np.mean(errorTest, axis = 0)
    # print('Parameters', global_params)
    print('Training Error', global_RmseTrain)
    print('Test Error', global_RmseTest)
    print('Time:', MPI.Wtime()-start_time)








