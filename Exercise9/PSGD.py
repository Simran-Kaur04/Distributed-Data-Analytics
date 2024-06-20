import torch
from torch.nn import Module
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

# Setting MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# CNN
class Network(Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(10, 30, kernel_size = 5)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.drop1 = nn.Dropout(0.15)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(120, 60)
        self.fc2 = nn.Linear(60, 10)
        self.soft = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        y = F.relu(self.pool1(self.conv1(x)))
        y = F.relu(self.pool2(self.conv2(y)))
        y = self.pool3(self.drop1(y))
        y = y.view(-1, 120)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        y = self.soft(y)
        return y

# Training the model
def train(train_data, epochs, criterion):
    model = Network()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    model.train()
    for epoch in range(epochs):
        for i, batch in enumerate(train_data):
            optimizer.zero_grad()
            yhat = model(batch[0])
            loss = criterion(yhat.squeeze(), batch[1].squeeze())
            loss.backward()
            optimizer.step() 
    return model

# Getting Accuracy for the Test data
def test(model, test_data, criterion):
    model.eval()
    correct_pred = 0
    with torch.no_grad():
        for input, label in test_data:
            yhat = model(input)
            pred = torch.max(yhat.data, 1)[1]
            correct_pred += (pred == label).sum().item()

    return 100. * correct_pred / len(test_data.dataset)



epochs = 20
criterion = nn.NLLLoss()

trainingSet = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
testSet = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor())

# Here sampler itself make partitions for the data to be sent on different workers
train_data = DataLoader(trainingSet, sampler = DistributedSampler(
            dataset=trainingSet,
            num_replicas=size,
            rank=rank
        ), batch_size=128)

test_data = DataLoader(testSet, batch_size = 128, shuffle = False)

start_time = MPI.Wtime()

model = train(train_data, epochs, criterion)


if rank != 0:
    model_params = []
    # parameters from all workers except master node are flattened and stored in a list
    for name, param in model.named_parameters():
        model_params.append(param.data.detach().numpy().flatten())

    comm.send(model_params,0)


if rank == 0:
    # parameters from the master node are extracted using get_parameters
    conv1W = model.get_parameter('conv1.weight').detach().numpy()
    conv1B = model.get_parameter('conv1.bias').detach().numpy()
    conv2W = model.get_parameter('conv2.weight').detach().numpy()
    conv2B = model.get_parameter('conv2.bias').detach().numpy()
    fc1W = model.get_parameter('fc1.weight').detach().numpy()
    fc1B = model.get_parameter('fc1.bias').detach().numpy()
    fc2W = model.get_parameter('fc2.weight').detach().numpy()
    fc2B = model.get_parameter('fc2.bias').detach().numpy()
    
    for i in range(1,size):
        model_params = comm.recv(source=i)

        # From all the workers the respective parameters are added to return average later
        conv1W = np.add(conv1W , np.array(model_params[0]).reshape([10, 1, 5, 5]))
        conv1B = np.add(conv1B , np.array(model_params[1]))
        conv2W = np.add(conv2W , np.array(model_params[2]).reshape([30, 10, 5, 5]))
        conv2B = np.add(conv2B , np.array(model_params[3]))
        fc1W = np.add(fc1W , np.array(model_params[4]).reshape([60, 120]))
        fc1B = np.add(fc1B , np.array(model_params[5]))
        fc2W = np.add(fc2W , np.array(model_params[6]).reshape([10, 60]))
        fc2B = np.add(fc2B , np.array(model_params[7]))

    conv1W /= size
    conv1B /= size
    conv2W /= size
    conv2B /= size
    fc1W /= size
    fc1B /= size
    fc2W /= size
    fc2B /= size

    #Updating model parameters
    # the model is tested using average parameters received from all workers
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == 'conv1.weight':
                param.copy_(torch.tensor(conv1W))
            if name == 'conv1.bias':
                param.copy_(torch.tensor(conv1B))
            if name == 'conv2.weight':
                param.copy_(torch.tensor(conv2W))
            if name == 'conv2.bias':
                param.copy_(torch.tensor(conv2B))
            if name == 'fc1.weight':
                param.copy_(torch.tensor(fc1W))
            if name == 'fc1.bias':
                param.copy_(torch.tensor(fc1B))
            if name == 'fc2.weight':
                param.copy_(torch.tensor(fc2W))
            if name == 'fc2.bias':
                param.copy_(torch.tensor(fc2B))

        print(f'----------Number of Processes {size}----------------')
        
        total_time_taken = MPI.Wtime() - start_time
        print(f'Total time taken for training: {total_time_taken}')
        
        accuracy = test(model, test_data, criterion)
        print(f'Accuracy on the Test Set : {accuracy}')

