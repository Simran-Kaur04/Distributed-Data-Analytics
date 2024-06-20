from mpi4py import MPI
import numpy as np

length = 10**3
u = np.zeros((length, 1))
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
step = length//size
startTime = MPI.Wtime()

if rank == 0:
    v = np.random.randint(10, size = (length, 1))
    w = np.random.randint(10, size = (length, 1))
    print('Vector V:',v)
    print('Vector W:',w)
    for j in range(step):
        u[j, 0] = v[j, 0] + w[j, 0]

    for i in range(1, size - 1):
        comm.send(v[i*step:(i + 1)*step, :], dest = i, tag = 1)
        comm.send(w[i*step:(i + 1)*step, :], dest = i, tag = 2 )
    comm.send(v[(size - 1)*step : , :], dest = (size - 1), tag = 1)
    comm.send(w[(size - 1)*step : , :], dest = (size - 1), tag = 2 )
    

    for i in range(1, size - 1):
        u[i*step:(i + 1)*step, :] = comm.recv(source = i, tag = 3)
    u[(size - 1)*step:, :] = comm.recv(source =(size - 1), tag = 3)
    
    print('Time:',MPI.Wtime()-startTime)
    print('Sum Vector U:',u)
        
else:
    v_new = comm.recv(source = 0, tag =1)
    w_new = comm.recv(source = 0, tag =2)
    c_new = np.zeros((len(v_new), 1))
    for k in range(len(v_new)):
        c_new[k, 0] = v_new[k, 0] + w_new[k, 0]
    comm.send(c_new, dest = 0, tag = 3)



