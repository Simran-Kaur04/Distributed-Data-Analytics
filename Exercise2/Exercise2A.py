from mpi4py import MPI
import numpy as np


length = 10**3
c = np.zeros((length, 1))
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
step = length//size
startTime = MPI.Wtime()

if rank == 0:
    A = np.random.randint(5, size = (length, length))
    b = np.random.randint(5, size = (length, 1))
    print('Matrix A::',A)
    print('Vector b::',b)
    for k in range(step):
        s = 0
        for l in range(length):
            s += A[k, l]*b[l, 0]
        c[k, :] = s

    for i in range(1, size - 1):
        comm.send(A[i*step:(i + 1)*step, :],dest = i,tag = 1)
        comm.send(b, dest = i,tag = 2)
    comm.send(A[(size - 1)*step: , :],dest = (size - 1), tag = 1)
    comm.send(b, dest = (size - 1), tag = 2)

    for i in range(1, size - 1):
        c[i*step: (i+1)*step, :] = comm.recv(source = i, tag = 3)
    c[(size - 1)*step:, :] = comm.recv(source = (size - 1), tag = 3)
    print('Resultant Vector C::',c)
    print('Time:',MPI.Wtime()-startTime)
        
else:
    A_chunk = comm.recv(source = 0, tag = 1)
    b_new = comm.recv(source = 0,tag = 2)
    c_new = np.zeros((len(A_chunk), 1))
    for k in range(len(A_chunk)):
        s = 0
        for l in range(len(b_new)):
            s += A_chunk[k, l]*b_new[l, 0]
        c_new[k, :] = s
    comm.send(c_new, dest = 0, tag = 3)