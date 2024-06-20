from mpi4py import MPI
import numpy as np


length = 4
C = np.zeros((length, length))
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
step = length//size
startTime = MPI.Wtime()

A = None
B = None
A_split = None

if rank == 0:
    A = np.random.randint(5, size = (length, length))
    B = np.random.randint(5, size = (length, length))
    print('matrix A:', A)
    print('matrix B:', B)
    A_new = A[: (size - 1)*step, :]
    A_split = []
    
    for i in range(size - 1):
        A_split.append(A_new[i*step:(i+1)*step,:])
    A_split.append(A[(size - 1)*step:, :])
    

B = comm.bcast(B, root = 0)
data = comm.scatter(A_split, root = 0)
for i in range(data.shape[0]):
    for j in range(B.shape[1]):
        s = 0
        for k in range(B.shape[0]):
            s += data[i, k]*B[k, j]
        data[i, j] = s
new_data = comm.gather(data, root = 0)
if rank == 0:
    C = np.vstack(new_data)
    print('matrix C:', C)
    print('Time:', MPI.Wtime()-startTime)

