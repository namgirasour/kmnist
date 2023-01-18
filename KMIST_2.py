#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:14:03 2022

@author: Bi
"""

#3 Hidden Layer Feedforward Neural Network (ReLU Activation)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
# import torch.nn.functional as F
import numpy as np

transform = transforms.Compose([
transforms.ColorJitter(contrast=1.5),
transforms.ToTensor(), 
])


train_dataset = datasets.MNIST(root='./data', 
                            train=True, 
                            transform=transform,
                            download=True)

test_dataset = datasets.MNIST(root='./data', 
                           train=False, 
                           transform=transform)

batch_size = 100
num_epochs = int(10)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

class FNN(nn.Module):
    def __init__(self, input_value, hidden_value, output_value):
        super(FNN, self).__init__()
        
        self.fc1 = nn.Linear(input_value, hidden_value) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_value, hidden_value)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_value, hidden_value)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_value, output_value)  

    def forward(self, x):
        
        out = self.fc1(x)
        
        out = self.relu1(out)

    
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

input_value = 28*28
hidden_value = 100
output_value = 10

model = FNN(input_value, hidden_value, output_value)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.view(-1, 28*28).requires_grad_()
        optimizer.zero_grad()
    
        outputs = model(images)
    
        loss = criterion(outputs, labels)

        loss.backward()
        
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
                     
            correct = 0
            total = 0
            predicted_list = []
            labels_list = []
            
            for images, labels in test_loader:
                
                images = images.view(-1, 28*28).requires_grad_()

                
                outputs = model(images)

                
                _, predicted = torch.max(outputs.data, 1)

                
                total += labels.size(0)

                
                correct += (predicted == labels).sum()
                predicted = predicted.detach().numpy()
                predicted_list = np.append(predicted_list, predicted)
                labels_list = np.append(labels_list, labels)
                recall = recall_score(predicted_list, labels_list, average='macro')
                print('Recall:', recall)
                precision = precision_score(predicted_list, labels_list, average='macro')
                print('Precision:', precision)

            accuracy = 100 * correct / total

            
            print('Accuracy: {}'.format(accuracy))