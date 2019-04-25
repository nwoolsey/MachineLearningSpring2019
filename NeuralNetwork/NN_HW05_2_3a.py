# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:35:31 2019

@author: nicho
"""

import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Hidden Layer Sizes
hls = np.array([3, 3])

x = np.array( [ [1],[1],[1] ])

ys = 1

num_attr = np.size(x)

#w1 = np.zeros((num_attr,hls[0]-1))
#w2 = np.zeros((hls[0],hls[1]-1))
#w3 = np.zeros((hls[1],1))

w1 = np.array([[-1, 1], [-2, 2], [-3, 3]])
w2 = np.array([[-1, 1], [-2, 2], [-3, 3]])
w3 = np.array([[-1],[2],[-1.5]])

z1 = np.zeros((hls[0],1))

z1[0] = 1

z2 = np.zeros((hls[1],1))

z2[0] = 1

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

print("Gradients of first layer:")
print(dL_dw1)
print("Gradients of second layer:")
print(dL_dw2)
print("Gradients of third layer:")
print(dL_dw3)

