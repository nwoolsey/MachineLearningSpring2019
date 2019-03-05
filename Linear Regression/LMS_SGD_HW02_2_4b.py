# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:58:46 2019

@author: nicho
"""

import numpy as np
import matplotlib.pyplot as plt


# Learning rate
r=float(1/(2**7))
print('Learning rate:', r)

#Tolerance threshold
tol = 1e-6

# Read training data
CSVfile = 'train_concrete.csv'

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
CSVfile = 'test_concrete.csv'

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
            
t = 0

w = np.zeros((num_attr+1,))

J = np.zeros((1,))

tol_acc = 0
#SGD
while t < 1e4 and tol_acc == 0 :
    
    i = np.random.randint(0,np.shape(train_data)[0])
    
#    for i in range(np.shape(train_data)[0]) :
    w_new = w + r * (train_data[i,num_attr] - np.dot(w,np.hstack((train_data[i,0:num_attr], 1))))*np.transpose(np.hstack((train_data[i,0:num_attr], 1)))
#    if np.linalg.norm(w_new-w) < tol:
#        tol_acc = 1
    w = w_new    
    if t > 0:
        J = np.vstack((J,0))
    for i in range(np.shape(train_data)[0]) :
        J[t] = J[t] + 0.5*(train_data[i,num_attr] - np.dot(w,np.hstack((train_data[i,0:num_attr], 1))) )**2
    t = t + 1
    
J_test = float(0)
for i in range(np.shape(test_data)[0]) :
    J_test = J_test + 0.5*(test_data[i,num_attr] - np.dot(w,np.hstack((test_data[i,0:num_attr], 1))) )**2

print('Learning rate: ', r)
print('weights: ', w)
print('Test Cost Function: ', J_test)
print('\n******************Please close the figure for the code to proceed******************\n') 
fig = plt.figure()
plt.plot(J)
plt.xlabel('Num. Iterations')
plt.ylabel('Cost Function (training), J')
plt.title('Stochastic Gradient Descent')
plt.show()
