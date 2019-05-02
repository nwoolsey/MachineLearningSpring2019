# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:58:46 2019

@author: nicho
"""

import numpy as np
import matplotlib.pyplot as plt


# Learning rate
r= 0.0001

# num. epoch per global step
T = 10

#num global steps
G = 100

# Noise Power
#N = np.array(([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10, 10]))
N = np.array(([0, 0, 0, 0, 0, 0, 0, 0]))

#num nodes
n = np.size(N)

# Perfect weights
real_w = np.array(([1, 1, 1, 1, 1, 1, 1, 0]))/np.sqrt(7)

# num. points per node training
P = 200
P_test = 2000

# number of atrributes (including bias term)
num_attr = np.size(real_w)

# Data 
train_data =  np.hstack( (2*np.random.rand(n*P,num_attr-1)-1,np.ones((n*P,1))))
test_data = np.hstack( (2*np.random.rand(P_test,num_attr-1)-1,np.ones((P_test,1))))

# labels
labels = np.sign(np.matmul(train_data,real_w))
labels_test = np.sign(np.matmul(test_data,real_w))

# starting weights
w_m1 = np.zeros((n,num_attr))
glob_w_m1 = np.zeros((1,num_attr))
w_m2 = np.zeros((n,num_attr))
glob_w_m2 = np.zeros((1,num_attr))
w_cumul = np.zeros((1,num_attr))
train_data_n= train_data

#Test Error Rate
#test_error = np.zeros((n,T*G))
test_error_glob_m1 = np.zeros((1,G))
test_error_glob_m2 = np.zeros((1,G))
test_error_glob_cumul = np.zeros((1,G))


# Added noise to training data
for i in range(n) :
    s1 = i*P
    s2 = (i+1)*P
    train_data_n[s1:s2,0:num_attr-1] = train_data[s1:s2,0:num_attr-1] + np.sqrt(N[i])*np.random.randn(P,num_attr-1)
    
for g in range(G):
    print(g)
    for t in range(T):
        perm = np.random.permutation(P*n)
        for i in range(n*P):
            x = train_data_n[perm[i],0:num_attr]
            y = labels[perm[i]]
            a = np.dot(w_cumul, train_data_n[perm[i],:])
            b = np.sign(a)
            if y*b <= 0:
                w_cumul = w_cumul + r*y*x
                w_cumul= w_cumul/np.linalg.norm(w_cumul)
    test_error_glob_cumul[0,g] = np.sum(np.sign(np.squeeze(np.matmul(test_data,np.transpose(w_cumul)))) != labels_test )/P_test
    for j in range(n):
        w_m1[j,:] = glob_w_m1
        w_m2[j,:] = glob_w_m2
        for t in range(T):
            perm = j*P + np.random.permutation(P)
            for i in range(P):
                x = train_data_n[perm[i],0:num_attr]
                y = labels[perm[i]]
                a = np.dot(w_m1[j,:], train_data_n[perm[i],:])
                b = np.sign(a)
                if y*b <= 0:
                    w_m1[j,:] = w_m1[j,:] + r*y*x
                    w_m1[j,:]= w_m1[j,:]/np.linalg.norm(w_m1[j,:])
                a = np.dot(w_m2[j,:], train_data_n[perm[i],:])
                b = np.sign(a)
                if y*b <= 0:
                    w_m2[j,:] = w_m2[j,:] + r*y*x
                    w_m2[j,:]= w_m2[j,:]/np.linalg.norm(w_m2[j,:])                    
#            test_error_m1[j,t+T*g] = np.sum(np.sign(np.matmul(test_data,np.transpose(w_m1[j,:]))) != labels_test )/P_test
#            test_error_m2[j,t+T*g] = np.sum(np.sign(np.matmul(test_data,np.transpose(w_m2[j,:]))) != labels_test )/P_test
    glob_w_m1 = np.mean(w_m1,axis=0)
    glob_w_m2 = np.median(w_m2,axis=0)
    test_error_glob_m1[0,g] = np.sum(np.sign(np.matmul(test_data,np.transpose(glob_w_m1))) != labels_test )/P_test
    test_error_glob_m2[0,g] = np.sum(np.sign(np.matmul(test_data,np.transpose(glob_w_m2))) != labels_test )/P_test
    

#plt.figure()
#plt.plot(np.transpose(w),'.',real_w,'.')

#plt.figure()
#plt.plot(np.transpose(test_error))

fig = plt.figure(0)
ax = plt.subplot(111)
ax.plot(test_error_glob_cumul[0,:],label='Centr.')
ax.plot(test_error_glob_m1[0,:],label='Decentr. (mean)')
ax.plot(test_error_glob_m2[0,:],label='Decentr. (median)')
ax.legend()
plt.xlabel('Num. Global Iterations')
plt.ylabel('Test Err.')
plt.show()