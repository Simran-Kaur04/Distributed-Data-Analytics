'''
    Lab Distributed Data Analytics Exercise 3
    Question 1 Distributed K-means Algorithm
'''
#Importing Packages
from mpi4py import MPI
import csv
import numpy as np
import random
import math

#Intializing the MPI Communicator
comm = MPI.COMM_WORLD

#The Rank of the running Process
rank = comm.Get_rank()

#Total worker Processes in our environment
size = comm.Get_size()

#Defining Root as Master node
root = 0

#Defining numbers of clusters
k = 6

#This function is used to calculate the euclidean distance between two points
def calculate_euclidean_distance(p,q):
    if len(p) != len(q):
        return Exception('Invalid Data point Dimensions')
    sum_squared = 0
    for i in range(len(p)):
        sum_squared += math.pow((p[i] - q[i]), 2)
    return math.sqrt(sum_squared)

#This function calculates mean point from different data points
def mean_of_datapoints(data_points):
    new_centroid = np.zeros(len(data_points))
    for i in range(len(data_points)):
        for j in range(len(data_points[i])):
            new_centroid[i] += data_points[i][j]
        new_centroid[i] /= len(data_points)
    return new_centroid

#Initializing different Variables
data = []
centroids = []
distance_matrix = None
membership = []

#Master node will read the file and distribute the data points among workers
if rank == 0:
    with open('Absenteeism_at_work_AAA/Absenteeism_at_work.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        next(csv_reader)  #Skipping the Header row
        for row in csv_reader:
            data.append([float(d) for d in row])
        
        #Populating the data points and different parameters
        data = np.array(data)
        centroids = [[random.randint(0, 400) for j in range(len(data[0]))] for i in range(k)]
        distance_matrix = np.zeros(shape=(len(data[:]),k))
        membership = np.zeros(shape=(len(data[:]),1))
        data = np.array_split(data,size,axis=0)

#Distributing Data among different Workers
dist_data = comm.scatter(data,root)
cluster_centroid = comm.bcast(centroids,root)
dist_mat = comm.bcast(distance_matrix,root)
membership_data = comm.bcast(membership,root)

wt = MPI.Wtime()

#Computing the Distance between data points and centroids
for i in range(len(dist_data)):
    data_index = (rank * len(dist_data)) + i
    for j in range(len(cluster_centroid)):
        distance = calculate_euclidean_distance(dist_data[i], cluster_centroid[j])
        dist_mat[data_index][j] = distance
    
    #Evaluating the Membership matrix
    member_index = np.argmin(dist_mat[data_index])
    membership_data[data_index][0] = member_index

#Waiting for all workers to complete
comm.Barrier()

#Master node will gather all the results from worker nodes
recv_buf = None
rec_buf_membership = None
if rank == 0:
    recv_buf = np.empty(shape=(len(dist_data)*k,k))
    recv_buf_membership = np.empty(shape=(len(dist_data)*k,1))

#Combining all workers distance matrix into final distance matrix
final_distance_matrix = comm.reduce(dist_mat, op = MPI.SUM, root = 0)

#Combining all workers membership matrix into final membership matrix
final_membership_matrix = comm.reduce(membership_data, op = MPI.SUM, root=0)

#Evaluating the time taken
wt = MPI.Wtime() - wt

if rank == 0: 
    print(f'Distance Matrix: {final_distance_matrix}')
    print(f'Total Time taken: {wt}')