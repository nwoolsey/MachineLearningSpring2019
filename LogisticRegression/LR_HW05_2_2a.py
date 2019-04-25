# -*- coding: utf-8 -*-
"""
Created on Thu Apr 01 21:58:46 2019

@author: nicho
"""

import numpy as np


# Learning rate
gamma0 = 0.001
d = 1
v = 0.01
print("v =",v)

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
            num_attr = np.size(data_row)-1
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

w = np.zeros((num_attr+1,))

num_train_exp = np.shape(train_data)[0]

num_test_exp = np.shape(test_data)[0]

for t in range(T):
    perm = np.random.permutation(num_train_exp)
    for i in range(num_train_exp):
        x = np.hstack((train_data[perm[i],0:num_attr], 1))
        y = 2*train_data[perm[i],-1] - 1
        e = np.exp(-y*np.dot(x,w))
        g = -y*x*e/(1+e) + w/v
        
        w = w - (gamma0/(1+gamma0*(t+1)/d))*g
            
    
print('Learned weights:',w)

train_error = 0;

for i in range(num_train_exp):
    x = np.hstack((train_data[i,0:num_attr], 1))
    y = 2*train_data[i,-1]-1
    a = np.dot(w, x )
    b = np.sign(a)
    if y*b <= 0:
        train_error = train_error + 1/num_train_exp
        
print('Tranin error:',train_error)


test_error = 0;

for i in range(num_test_exp):
    x = np.hstack((test_data[i,0:num_attr], 1))
    y = 2*test_data[i,-1]-1
    a = np.dot(w, x )
    b = np.sign(a)
    if y*b <= 0:
        test_error = test_error + 1/num_test_exp
        
print('Test error:',test_error)