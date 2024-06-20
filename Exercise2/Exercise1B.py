from mpi4py import MPI
import numpy as np


length = 10**5
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
step = length//size
u = np.zeros((size, 1))
avg_vector = 0
startTime = MPI.Wtime()

if rank == 0:
    v = np.random.randint(5, size = (length, 1))
    print('Vector V:',v)
    s = 0
    for j in range(step):
        s += v[j, 0]
    
    u[0, 0] = s/step
    

    for i in range(1, size - 1):
        comm.send(v[i*step:(i + 1)*step, :],dest = i, tag = 1)
    comm.send(v[(size - 1)*step : , :], dest = (size - 1), tag = 1)
    

    for i in range(1, size):
        u[i, :] = comm.recv(source = i, tag = 2)
    
    a = 0
    for l in range(size):
        a += u[l, 0]
    avg_vector = a/size

    print('Average:',avg_vector)
    print('Time:',MPI.Wtime()-startTime)
        
else:
    v_new = comm.recv(source = 0,tag =1)
    c_new = 0
    for k in range(len(v_new)):
        c_new += v_new[k, 0]
    avg = c_new/len(v_new)
    comm.send(avg, dest = 0, tag = 2)