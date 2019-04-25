# -*- coding: utf-8 -*-
"""
Created on Thu Apr 01 21:58:46 2019

@author: nicho
"""

import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Learning rate
gamma0 = 0.1
d = 1
layer_size = 50
print('Layer size:',layer_size)



# num. epoch
T = 100

# Read training data
CSVfile = 'train.csv'

c=0
with open (CSVfile , 'r') as f :
    for line in f :
        
        terms = line.strip().split(',')
        data_row = np.zeros(  (  1,np.size(terms)  )  )
        for i in range(np.size(terms)):
            data_row[0,i] = float(terms[i])
        if c == 0:
            train_data = data_row
            c=c+1
            num_attr = np.size(data_row)
        else :
            train_data = np.vstack((train_data,data_row))
            
# Read test data
CSVfile = 'test.csv'

c=0
with open (CSVfile , 'r') as f :
    for line in f :
        
        terms = line.strip().split(',')
        data_row = np.zeros(  (  1,np.size(terms)  )  )
        for i in range(np.size(terms)):
            data_row[0,i] = float(terms[i])
        if c == 0:
            test_data = data_row
            c=c+1
        else :
            test_data = np.vstack((test_data,data_row))

num_train_exp = np.shape(train_data)[0]

num_test_exp = np.shape(test_data)[0]

hls = np.array([layer_size, layer_size])

w1 = np.zeros((num_attr,hls[0]-1))
w2 = np.zeros((hls[0],hls[1]-1))
w3 = np.zeros((hls[1],1))

z1 = np.zeros((hls[0],1))

z1[0] = 1

z2 = np.zeros((hls[1],1))

z2[0] = 1

for t in range(T):
    perm = np.random.permutation(num_train_exp)
    for i in range(num_train_exp):
        x = np.reshape(np.hstack((train_data[perm[i],0:num_attr-1], 1)),(num_attr,1))
        ys = 2*train_data[perm[i],-1] - 1
        
        # Forward Pass
        z1[1:hls[0],:] = sigmoid(np.matmul(np.transpose(w1),x))
        z2[1:hls[1],:] = sigmoid(np.matmul(np.transpose(w2),z1))
        y = np.matmul(np.transpose(w3),z2)
        
        # Back Propagation
        dL_dy = y - ys
        dL_dw3 = dL_dy*z2
        dL_dz2 = dL_dy*w3
        dz2_dw2 = np.matmul(z1,np.transpose(z2*(1-z2)))
        dL_dw2 = dz2_dw2[:,1:hls[0]]*np.repeat(np.transpose(dL_dz2[1:hls[1]]),hls[0],axis=0)
        dz2_dz1 = w2*np.transpose(z2[1:hls[1]]*(1-z2[1:hls[1]]))
        dL_dz1 = np.reshape(np.sum(dz2_dz1*np.repeat(np.transpose(dL_dz2[1:hls[1]]),hls[0],axis=0),axis=1),(hls[0],1))
        dz1_dw1 = np.matmul(x,np.transpose(z1*(1-z1)))
        dL_dw1 = dz1_dw1[:,1:hls[0]]*np.repeat(np.transpose(dL_dz1[1:hls[0]]),num_attr,axis=0)
        
        #Weight Updates
        gma = (gamma0/(1+gamma0*(t+1)/d))
        w1 = w1 - gma*dL_dw1
        w2 = w2 - gma*dL_dw2
        w3 = w3 - gma*dL_dw3
            


train_error = 0;

for i in range(num_train_exp):
    x = np.reshape(np.hstack((train_data[i,0:num_attr-1], 1)),(num_attr,1))
    ys = 2*train_data[i,-1] - 1
    
    # Forward Pass
    z1[1:hls[0],:] = sigmoid(np.matmul(np.transpose(w1),x))
    z2[1:hls[1],:] = sigmoid(np.matmul(np.transpose(w2),z1))
    y = np.matmul(np.transpose(w3),z2)
    if np.sign(y)*np.sign(ys) <= 0:
        train_error = train_error + 1/num_train_exp
        
print('Train error:',train_error)


test_error = 0;

for i in range(num_test_exp):
    x = np.reshape(np.hstack((test_data[i,0:num_attr-1], 1)),(num_attr,1))
    ys = 2*test_data[i,-1] - 1
    
    # Forward Pass
    z1[1:hls[0],:] = sigmoid(np.matmul(np.transpose(w1),x))
    z2[1:hls[1],:] = sigmoid(np.matmul(np.transpose(w2),z1))
    y = np.matmul(np.transpose(w3),z2)
    if np.sign(y)*np.sign(ys) <= 0:
        test_error = test_error + 1/num_test_exp
        
print('Test error:',test_error)