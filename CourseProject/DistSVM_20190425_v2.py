# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:58:46 2019

@author: nicho
"""

import numpy as np
import matplotlib.pyplot as plt


# Learning rate
r = 0.00001
C = 250/783
gamma0 = 0.0005
d = gamma0
print(gamma0)
print(C)

# num. epoch per global step
T = 1

#num global steps
G = 1000

#num nodes
n = np.array([1,2,5,10,20,50,100])

# Perfect weights
real_w = np.array(([1, 1, 1, 1, 1, 1, 1, 0]))/np.sqrt(7)

# num. points
P = 500
print(P)
P_test = 5000

# number of atrributes (including bias term)
num_attr = np.size(real_w)

# Data 
train_data =  np.hstack( (2*np.random.rand(P,num_attr-1)-1,np.ones((P,1))))
#test_data = np.hstack( (2*np.random.rand(P_test,num_attr-1)-1,np.ones((P_test,1))))

# labels
labels = np.sign(np.matmul(train_data,real_w))
#labels_test = np.sign(np.matmul(test_data,real_w))

# starting weights
w = np.zeros((np.max(n),num_attr))
w_glob = np.zeros((np.size(n),num_attr))



LossFunc = np.zeros((np.size(n),G))


for k in range(np.size(n)):    
    print("=====")
    print(n[k])
    for g in range(G):
        if np.remainder(g,100)==0:
            print(g)
        for j in range(n[k]):
            w[j,:] = w_glob[k,:]
            for t in range(T):
                perm = int(j*(P/n[k])) + np.random.permutation(int(P/n[k]))
                for i in range(int(P/n[k])):
                    x = train_data[perm[i],0:num_attr]
                    y = labels[perm[i]]
                    a = np.dot(w[j,:], x )
                    dJ = np.hstack(([w[j,0:num_attr-1],0]))
                    if 1 - y*a > 0:
                        dJ = dJ - C * P * y * x
                    w[j,:] = w[j,:] - (gamma0/(1+gamma0*(t+T*g + 1)/d))*dJ

                   
    
        w_glob[k,:] = np.mean(w[0:n[k],:],axis=0)
    #    test_error_glob_m1[0,g] = np.sum(np.sign(np.matmul(test_data,np.transpose(glob_w_m1))) != labels_test )/P_test
    #    test_error_glob_m2[0,g] = np.sum(np.sign(np.matmul(test_data,np.transpose(glob_w_m2))) != labels_test )/P_test
        S = 1-labels*np.squeeze(np.matmul(train_data,np.transpose(w_glob[k,:])))
        LossFunc[k,g] = 0.5*(np.linalg.norm(w_glob[k,:]) - w_glob[k,-1]**2 ) + C*np.sum(S*(S>0))

#fig = plt.figure(0)
#ax = plt.subplot(111)
#ax.plot(test_error_glob_cumul[0,:],label='Centr.')
#ax.plot(test_error_glob_m1[0,:],label='Decentr. (mean)')
#ax.plot(test_error_glob_m2[0,:],label='Decentr. (median)')
#ax.legend()
#plt.xlabel('Num. Global Iterations')
#plt.ylabel('Test Err.')
#plt.show()

fig = plt.figure(1)
ax = plt.subplot(111)
for k in range(np.size(n)):
    ax.plot(LossFunc[k,:],label= n[k] )
ax.legend()
plt.xlabel('Num. Global Iterations')
plt.ylabel('Loss Func.')
plt.show()