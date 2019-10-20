#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:06:55 2019

@author: AlexandrosTzikas
"""
import torch
import torch.nn as nn

#Load data
from numpy import genfromtxt
all_data = genfromtxt('/Users/AlexandrosTzikas/Desktop/DiplomaThesis/training3-2.csv', dtype=float, delimiter=';', max_rows=200)
train_data_x=all_data[0:100, 0:3]
train_data_y=all_data[0:100, 3:]
test_data_x=all_data[100:200, 0:3]
test_data_y=all_data[100:200, 3:]

#2D array of data to tensor
trainT_data_x=torch.from_numpy(train_data_x)
trainT_data_y=torch.from_numpy(train_data_y)
testT_data_x=torch.from_numpy(test_data_x)
testT_data_y=torch.from_numpy(test_data_y)



# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h_1, n_h_2, n_out= 3, 20, 30, 22

# Create a model
model = nn.Sequential(nn.Linear(n_in, n_h_1), #First (All features) to Second layer Connections 
                      nn.Tanh(), 
                      nn.Linear(n_h_1, n_h_2), 
                      nn.Tanh(),
                      nn.Linear(n_h_2, n_out), 
                      nn.Tanh())

#Construct the loss function
criterion = torch.nn.MSELoss()

# Construct the optimizer (Stochastic Gradient Descent in this case) 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model=model.float()

for epoch in range(50):
    # Forward pass: Compute predicted y by passing x to the model 
    y_pred = model(trainT_data_x.float())
    # Compute and print loss
    loss = criterion(y_pred, trainT_data_y.float())
    print('epoch:', epoch,'loss:', loss.item())
    # Zero the gradients
    optimizer.zero_grad()
    # perform a backward pass (backpropagation) - compute gradients
    loss.backward()
    # Update the parameters
    optimizer.step()
    
print(criterion(model(trainT_data_x.float()), trainT_data_y.float()))
print(criterion(model(testT_data_x.float()), trainT_data_y.float()))