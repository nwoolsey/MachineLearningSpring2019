# -*- coding: utf-8 -*-
"""
Created on Thu Apr 01 21:58:46 2019

@author: nicho
"""

from scipy.optimize import minimize
import numpy as np
#import matplotlib.pyplot as plt


# Learning rate
C = 100/783
#gamma0 = 0.01
#d = 0.01

# num. epoch
#T = 100

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

x =  np.hstack((train_data[:,0:-1], np.ones((num_train_exp,1)) ))

y = 2*train_data[:,-1]-1

# Optimize

X = np.zeros((num_train_exp,num_train_exp))
for i in range(num_train_exp) :
        for j in range(num_train_exp) :
            X[i,j]=np.dot(x[i,:],x[j,:])
            
a0 = np.zeros((num_train_exp,))

def objective(a):
    a1 = np.reshape(np.multiply(a,y),[num_train_exp,1])

    A = np.matmul(a1,np.transpose(a1))
#    A = np.tile(np.multiply(y,a),[num_train_exp,1])
    
#    B = np.multiply(np.transpose(A),np.multiply(A,X))
    B = np.multiply(A,X)
    F = 0.5*B.sum() - np.sum(a)
        
    return F

def constraint1(a):
    return np.dot(a,y)

con1 = {'type': 'eq', 'fun': constraint1}

bnds = ((0, C),) * num_train_exp

cons = [con1]

#print(objective(a0))

sol = minimize(objective, a0, method = 'SLSQP', bounds=bnds, constraints=cons)

a = sol.x

w = np.zeros((num_attr+1,))    

for i in range(num_train_exp):
    x = np.hstack((train_data[i,0:num_attr], 1))
    w = w + a[i]*y[i]*x

   
    
print('Learned weights:',w)

train_error = 0;

for i in range(num_train_exp):
    x = np.hstack((train_data[i,0:num_attr], 1))
    y = 2*train_data[i,-1]-1
    a = np.dot(w, x )
    b = np.sign(a)
    if y*b <= 0:
        train_error = train_error + 1/num_train_exp
        
print('Train error:',train_error)


test_error = 0;

for i in range(num_test_exp):
    x = np.hstack((test_data[i,0:num_attr], 1))
    y = 2*test_data[i,-1]-1
    a = np.dot(w, x )
    b = np.sign(a)
    if y*b <= 0:
        test_error = test_error + 1/num_test_exp
        
print('Test error:',test_error)