# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 20:16:25 2019

@author: nicho
"""

import numpy as np
import matplotlib.pyplot as plt

# Calculate entropy given a vector of (integer) labels, L, and number of unique labels, n
def Entr_hw2(L,n,W) :
    m = np.sum(W)
    H = float(0)
    if m > 0 :
        p = 0
        for i in range(n):
            p = np.sum(W[np.where((L==i))[0]])/ float(m)
            if p > 0 :
                H = H - p*np.log2(p)
    return H

# Calculate ME
def ME_hw2(L,n,W):
    m = np.sum(W)
    ME = float(0)
    maxp = float(0)
    if m > 0 :
        p = 0
        for i in range(n):
            p = np.sum(W[np.where((L==i))[0]])/ float(m)
            if p > maxp :
                maxp = p
    ME = 1-maxp
    return ME

#Calaculate GI
def GI_hw2(L,n,W):
    m = np.sum(W)
    GI = float(1)
    if m > 0 :
        p = 0
        for i in range(n):
            p = np.sum(W[np.where((L==i))[0]])/ float(m)
            GI = GI - p**2
    return GI

# Recursive ID3 algorithm
def ID3_hw2(labels,label_size,allterms,attr_size,attr_active,MeasType,TreeLoc,TreeDepth,next_branch,weights):
    
    # Calculate measure of all inputted labels
    if MeasType == 'Entropy' :
        M = Entr_hw2(labels,label_size,weights)
    elif MeasType == 'ME' :
        M = ME_hw2(labels,label_size,weights)
    else :
        M = GI_hw2(labels,label_size,weights)
    
    # Conditional Measurement variable initialization
    CM = np.zeros((np.size(attr_size),))
    
    # total number of examples
#    num_exp = float(np.size(labels))
    
    # Initializing Information Gain variable
    InfoGain = float(-1);
    
    # Variable to decide which attribute will split the tree next
    SplitAttr = -1
    
    # Cycle through all attributes
    for i in attr_active:
        CM = 0
        # cycle through all values for a specific atrribute to compute InfoGain
        for j in range(attr_size[i]):
            # find examples with specific attribute value
            locs = np.where((allterms[:,i]==j))[0]
            L = labels[locs]
            W = weights[locs]
            if MeasType == 'Entropy' :
                cm = Entr_hw2(L,label_size,W)
            elif MeasType == 'ME' :
                cm = ME_hw2(L,label_size,W)
            else :
                cm = GI_hw2(L,label_size,W)
            CM = CM + cm*np.sum(W)
        InfoGain_temp = M-CM
        # Keep track of largest InfoGain
        if InfoGain_temp > InfoGain :
            InfoGain = InfoGain_temp
            SplitAttr = i
    
    # Tracks the branches of the tree. Each row represents 1 tree node. The
    # values along the row represent either the index of the next node or
    # if it is negative it represents a leaf node
    branches_tree = np.zeros((1,np.max(attr_size)))
    
    # Keeps track of which atrribute splits the branches form each node
    attr_tree =[ [SplitAttr  ] [-1]]
    
    # Removes atrribute used to split the data form the "active" or "unused"
    # set of attributes. Gets passed on to future ID3 calls
    attr_active = np.delete(attr_active,np.where(attr_active == SplitAttr)[0][0])
    
    
    # Find most common label
    p = np.zeros((label_size,), dtype = float)
    total_weight = np.sum(weights)
    for i in range(label_size):
        p[i] = np.sum(weights[np.where((labels==i))[0]])/total_weight
    mc_p = np.argmax(p)
    
    
    # Cycles through all values of the splitting atrribute
    for j in range(attr_size[SplitAttr]) :
        # finds all examples with given attribute
        locs = np.where((allterms[:,SplitAttr]==j))[0]
        L = labels[locs]
        W = weights[locs]
        
        
        
        
        # if there are no exmaples OR there is only 1 unique label OR
        # the tree depth has been reached OR there are no remaining active
        # (unused) attributes, a leaf node is defined with the most common
        # label
        
        if np.size(L) == 0:
            branches_tree[0,j] = -1*mc_p-1
        elif np.size(np.unique(L)) == 1 or TreeLoc == TreeDepth or np.size(attr_active) == 0 :
            # Most common label within subset
            p = np.zeros((label_size,), dtype = float)
            total_weight = np.sum(W)
            for i in range(label_size):
                p[i] = np.sum(W[np.where((L==i))[0]])/total_weight
            mc_p_ss = np.argmax(p)
            branches_tree[0,j] = -1*mc_p_ss-1
        # Otherwise, ID3 is called again to define more branches
        else :
            next_branch = next_branch + 1
            branches_tree[0,j] = next_branch
            AT, BT, next_branch = ID3_hw2(L,label_size,allterms[locs,:],attr_size,attr_active,MeasType,TreeLoc+1,TreeDepth,next_branch,W)
            attr_tree = np.vstack((attr_tree,AT))
            branches_tree = np.vstack((branches_tree,BT))
            
    
    return attr_tree, branches_tree, next_branch


def TreeOutcome_hw2(terms,branches_tree,attr_tree,num_exp) :

    tree_outcome = np.zeros((num_exp,),dtype = int)
    
    for i in range(num_exp) :
        tree_outcome[i] = branches_tree[0,terms[i,attr_tree[0]]]
        while tree_outcome[i] >= 0:
            tree_outcome[i] = branches_tree[tree_outcome[i],terms[i,attr_tree[tree_outcome[i]]]]
        tree_outcome[i] = -1*(tree_outcome[i]+1)
    
    return tree_outcome

# MAIN FILE
    
# CHANGE AS NECESSARY
    
TreeDepth = 2
MeasType = 'Entropy'
num_iter = 1000

print('TreeDepth:', TreeDepth)
print('Measurement Type:',MeasType)
print('Number of Trees:',num_iter)
print('This may take a few minutes...')

# Read training data
CSVfile = 'train2.csv'

#Numerical attributes
numerical_attr = np.array([0,5,9,11,12,13,14],dtype = int)

# Read training data
c = 0
with open (CSVfile , 'r') as f :
    for line in f :
        
        terms = line.strip().split(',')
        if c == 0:
            # number of attributes
            num_attr = np.shape(terms)[0]-1
            # Each unique value is assigned an integer value
            attr_vals = np.vstack((terms[0:num_attr],num_attr*[''])).astype('U15')
            # Each unique label is assigned an integer value
            label_vals = terms[-1]
            # Number of unique labels
            label_size = 1
            # Tracks labels of the examples
            labels = 0
            # Ass0ciated index of the label value
            label_index = 1
            # Tracks number of unique value sof the atrributes
            attr_size =  np.ones((num_attr,),dtype = int)
            # Tracks atrribute value sof the examples
            allterms = np.zeros((num_attr,),dtype = int)
            #
            term_arr = np.array(terms,dtype = 'U15')
            allterms[numerical_attr] = term_arr[numerical_attr]
            # Associated index of the attribute value
            terms_index = np.zeros((num_attr,),dtype = int)
        else:
            # Looks for new labels
            if terms[-1] not in label_vals:
                label_vals = np.vstack((label_vals,terms[-1]))
                label_size = label_size + 1
            label_index = np.where(terms[-1]==label_vals)[0]
            labels = np.vstack((labels,label_index))
            # cycles through attributes
            for i in range(num_attr):
                # Looks for new atrribute values
                if (terms[i] not in attr_vals[0:attr_size[i],i]) and (i not in numerical_attr) :
                    attr_size[i] = attr_size[i] + 1
                    if attr_size[i]  > np.shape(attr_vals)[0]:
                        attr_vals = np.vstack((attr_vals,num_attr*['']))
                    attr_vals[attr_size[i]-1,i] = terms[i]
                if i not in numerical_attr :
                    terms_index[i] = np.where(terms[i]==attr_vals[0:attr_size[i]+1,i])[0][0]
                else :
                    terms_index[i] = int(terms[i])
            allterms = np.vstack((allterms,terms_index))
            
        c = c+1
c=0
med = np.zeros((np.size(numerical_attr),),dtype = float)
for i in numerical_attr :
    med[c] = np.median(allterms[:,i])
    attr_size[i]=2
    attr_vals[0:2,i] = [0,1]
    allterms[:,i] = allterms[:,i] > med[c]
    c=c+1

num_exp = np.shape(allterms)[0]

# training weights
weights = np.zeros((num_exp,)) + 1/num_exp

# Read test data
CSVfile = 'test2.csv'
allterms_test = np.zeros((0,num_attr),dtype = int)
labels_test = np.zeros((0,1),dtype = int)
with open (CSVfile , 'r') as f :
    for line in f :
        terms = line.strip().split(',')
        if terms[-1] not in label_vals:
            label_vals = np.vstack((label_vals,terms[-1]))
            label_size = label_size + 1
        label_index = np.where(terms[-1]==label_vals)[0]
        labels_test = np.vstack((labels_test,label_index))
        for i in range(num_attr):
            if (terms[i] not in attr_vals[0:attr_size[i],i]) and (i not in numerical_attr) :
                attr_size[i] = attr_size[i] + 1
                if attr_size[i]  > np.shape(attr_vals)[0]:
                    attr_vals = np.vstack((attr_vals,num_attr*['']))
                attr_vals[attr_size[i]-1,i] = terms[i]
            if i not in numerical_attr :
                terms_index[i] = np.where(terms[i]==attr_vals[0:attr_size[i]+1,i])[0][0]
            else :
                terms_index[i] = int(terms[i]) > med[np.where(i == numerical_attr)[0][0]]
        allterms_test = np.vstack((allterms_test,terms_index))

# Total number of test examples     
num_exp_test = np.shape(allterms_test)[0]


#Initialize boost outcomes
boost_outcome = np.zeros((num_exp,),dtype = float)
boost_outcome_test = np.zeros((num_exp_test,),dtype = float)

# Active atrributes. All are active to start
attr_active = np.arange(num_attr)
    
train_error = np.zeros((num_iter,), dtype = float)

test_error = np.zeros((num_iter,), dtype = float)

iteration_train_error = np.zeros((num_iter,), dtype = float)

iteration_test_error = np.zeros((num_iter,), dtype = float)


T = np.arange(1,num_iter+1)

for i in range(num_iter):
    
    # input into ID3, defines where in the tree it is (starts at 0)
    next_branch = 0
    
    
    attr_tree, branches_tree, next_branch = ID3_hw2(labels,label_size,allterms,attr_size,attr_active,MeasType,1,TreeDepth,next_branch,weights)
    
    ## Training Error
    
    tree_outcome = TreeOutcome_hw2(allterms,branches_tree,attr_tree,num_exp)
    
    epsilon = np.sum(weights[np.where(tree_outcome != np.squeeze(labels))[0]])
    
#    print('E:',epsilon)
    
    alpha = 0.5*np.log((1-epsilon)/epsilon)
    
#    print('a:',alpha)
    
    boost_outcome = boost_outcome + alpha*(2*tree_outcome-1)
    
    train_error[i] = float(np.sum( (boost_outcome > 0) != np.squeeze(labels) ))/float(num_exp)
#    print('Training Error:', train_error)
    
    iteration_train_error[i] = float(np.sum( tree_outcome != np.squeeze(labels) ))/float(num_exp)
    
    weights = np.multiply(weights,np.exp(alpha*(2*( tree_outcome  != np.squeeze(labels) )-1)))
    
    weights = weights/np.sum(weights)
    
    ## Test Data Error
    
    # Define variable ot keep track of tree outcome
    tree_outcome_test = TreeOutcome_hw2(allterms_test,branches_tree,attr_tree,num_exp_test)
    
    boost_outcome_test = boost_outcome_test + alpha*(2*tree_outcome_test-1)
    
    test_error[i] = float(np.sum( (boost_outcome_test > 0) != np.squeeze(labels_test) ))/float(num_exp_test)
    
    iteration_test_error[i] = float(np.sum( tree_outcome_test != np.squeeze(labels_test) ))/float(num_exp_test)
    
#    print('Test Error:', test_error)

print('\n******************Please close the figure for the code to proceed******************\n') 
fig = plt.figure(1)
ax = plt.subplot(111)
ax.plot(T,test_error,label='Test Er.')
ax.plot(T,train_error,label='Train Er.')
ax.legend()
plt.title('AdaBoost Error')
plt.xlabel('Number of Stumps')
plt.show()


print('\n******************Please close the figure for the code to proceed******************\n') 
fig = plt.figure(2)
ax = plt.subplot(111)
ax.plot(T,iteration_test_error,'.',label='Test Er.')
ax.plot(T,iteration_train_error,'.',label='Train Er.')
ax.legend()
plt.title('Individual Stump Error')
plt.xlabel('Stump Number')
plt.show()
