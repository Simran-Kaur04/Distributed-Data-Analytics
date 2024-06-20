import torch
import torch.nn.functional as F
from torch.nn import Module
from torchvision import datasets,transforms
from torch import nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler
import time

# Network
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
def train(model, criterion, training_loader, worker_rank, epochs):

    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    model.train()

    for _ in range(epochs):
        for batch_idx, (input, label) in enumerate(training_loader):
            optimizer.zero_grad()
            yhat = model(input)
            loss = criterion(yhat.squeeze(), label.squeeze())
            loss.backward(retain_graph = True)
            optimizer.step()
    
    return model

# Testing the model accuracy on the test set
def test(model, criterion, test_data):
    model.eval()
    correct_pred = 0
    with torch.no_grad():
        for input, label in test_data:
            output = model(input)
            pred = torch.max(output.data, 1)[1]
            correct_pred += (pred == label).sum().item()

    print(f'Accuracy : {100. * correct_pred / len(test_data.dataset)}')

if __name__ == '__main__':

    Total_workers = 5
    epochs = 20
    mp.set_start_method('spawn')
    model = Network()
    model.share_memory()

    criterion = nn.NLLLoss()

    trainingSet = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
    testSet = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor())


    test_data = DataLoader(testSet, batch_size = 128, shuffle=False)


    processes = []
    start_time = time.perf_counter()
    for i in range(Total_workers):

        train_data = torch.utils.data.DataLoader(trainingSet, sampler=DistributedSampler(
                dataset = trainingSet,
                num_replicas = Total_workers,
                rank = i
            ), batch_size = 128)
        
        p = mp.Process(target = train, args=(model, criterion, train_data, i, epochs))
        p.start()
        processes.append(p)

    for p in processes: 
        p.join()

    total_time_taken = time.perf_counter() - start_time

    print(f'----------Number of Processes {Total_workers}----------------')

    print(f'Total time taken for training: {total_time_taken}')

    test(model, criterion, test_data)


