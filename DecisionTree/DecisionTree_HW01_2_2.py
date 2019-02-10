# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 20:16:25 2019

@author: nicho
"""

import numpy as np

# Calculate entropy given a vector of (integer) labels, L, and number of unique labels, n
def Entr_hw1(L,n) :
    m = np.shape(L)[0]
    H = float(0)
    if m > 0 :
        for i in range(n):
            p = float(np.size(np.where((L==i))[0]))/ float(m)
            if p > 0 :
                H = H - p*np.log2(p)
    return H

# Calculate ME
def ME_hw1(L,n):
    m = np.shape(L)[0]
    ME = float(0)
    maxp = float(0)
    if m > 0 :
        for i in range(n):
            p = float(np.size(np.where((L==i))[0]))/ float(m)
            if p > maxp :
                maxp = p
    ME = 1-maxp
    return ME

#Calaculate GI
def GI_hw1(L,n):
    m = np.shape(L)[0]
    GI = float(1)
    if m > 0 :
        for i in range(n):
            p = float(np.size(np.where((L==i))[0]))/ float(m)
            GI = GI - p**2
    return GI

# Recursive ID3 algorithm
def ID3_hw1(labels,label_size,allterms,attr_size,attr_active,MeasType,TreeLoc,TreeDepth,next_branch):
    
    # Calculate measure of all inputted labels
    if MeasType == 'Entropy' :
        M = Entr_hw1(labels,label_size)
    elif MeasType == 'ME' :
        M = ME_hw1(labels,label_size)
    else :
        M = GI_hw1(labels,label_size)
    
    # Conditional Measurement variable initialization
    CM = np.zeros((np.size(attr_size),))
    
    # total number of examples
    num_exp = float(np.size(labels))
    
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
            if MeasType == 'Entropy' :
                cm = Entr_hw1(L,label_size)
            elif MeasType == 'ME' :
                cm = ME_hw1(L,label_size)
            else :
                cm = GI_hw1(L,label_size)
            CM = CM + cm*float(np.size(locs))/num_exp
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
    
    # Cycles through all values of the splitting atrribute
    for j in range(attr_size[SplitAttr]) :
        # finds all examples with given attribute
        locs = np.where((allterms[:,SplitAttr]==j))[0]
        L = labels[locs]
        # if there are no exmaples OR there is only 1 unique label OR
        # the tree depth has been reached OR there are no remaining active
        # (unused) attributes, a leaf node is defined with the most common
        # label
        if np.size(L) == 0:
            (values,counts) = np.unique(labels,return_counts=True)
            branches_tree[0,j] = -1*values[np.argmax(counts)]-1
        elif np.size(np.unique(L)) == 1 or TreeLoc == TreeDepth or np.size(attr_active) == 0 :
            (values,counts) = np.unique(L,return_counts=True)
            branches_tree[0,j] = -1*values[np.argmax(counts)]-1
        # Otherwise, ID3 is called again to define more branches
        else :
            next_branch = next_branch + 1
            branches_tree[0,j] = next_branch
            AT, BT, next_branch = ID3_hw1(L,label_size,allterms[locs,:],attr_size,attr_active,MeasType,TreeLoc+1,TreeDepth,next_branch)
            attr_tree = np.vstack((attr_tree,AT))
            branches_tree = np.vstack((branches_tree,BT))
            
    
    return attr_tree, branches_tree, next_branch

# Main File
    
# CHANGE AS NECESSARY
#
TreeDepth = 6
MeasType = 'GI'

print('TreeDepth:', TreeDepth)
print('Measurement Type:',MeasType)

# Read training data
CSVfile = 'train1.csv'

c = 0
with open (CSVfile , 'r') as f :
    for line in f :
        
        terms = line.strip().split(',')
        if c == 0:
            # number of attributes
            num_attr = np.shape(terms)[0]-1
            # Each unique value is assigned an integer value
            attr_vals = np.vstack((terms[0:num_attr],num_attr*['']))
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
                if terms[i] not in attr_vals[0:attr_size[i],i] :
                    attr_size[i] = attr_size[i] + 1
                    if attr_size[i]  > np.shape(attr_vals)[0]:
                        attr_vals = np.vstack((attr_vals,num_attr*['']))
                    attr_vals[attr_size[i]-1,i] = terms[i]
                terms_index[i] = np.where(terms[i]==attr_vals[0:attr_size[i]+1,i])[0][0]
            allterms = np.vstack((allterms,terms_index))
            
        c = c+1

# input into ID3, defines where in the tree it is (starts at 0)
next_branch = 0

attr_active = np.arange(num_attr)

attr_tree, branches_tree, next_branch = ID3_hw1(labels,label_size,allterms,attr_size,attr_active,MeasType,1,TreeDepth,next_branch)

## Training Error

num_exp = np.shape(allterms)[0]
tree_outcome = np.zeros((num_exp,),dtype = int)
error_count = 0;

for i in range(num_exp) :
    tree_outcome[i] = branches_tree[0,allterms[i,attr_tree[0]]]
    while tree_outcome[i] >= 0:
        tree_outcome[i] = branches_tree[tree_outcome[i],allterms[i,attr_tree[tree_outcome[i]]]]
    tree_outcome[i] = -1*(tree_outcome[i]+1)
    if tree_outcome[i] != labels[i]:
        error_count = error_count + 1
        
train_error = float(error_count)/float(num_exp)
print('Training Error:', train_error)

## Test Data Error
CSVfile = 'test1.csv'

# Read test data
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
            if terms[i] not in attr_vals[0:attr_size[i],i] :
                attr_size[i] = attr_size[i] + 1
                if attr_size[i]  > np.shape(attr_vals)[0]:
                    attr_vals = np.vstack((attr_vals,num_attr*['']))
                attr_vals[attr_size[i]-1,i] = terms[i]
            terms_index[i] = np.where(terms[i]==attr_vals[0:attr_size[i]+1,i])[0][0]
        allterms_test = np.vstack((allterms_test,terms_index))

# Total number of test examples     
num_exp = np.shape(allterms_test)[0]
# Define variable ot keep track of tree outcome
tree_outcome_test = np.zeros((num_exp,),dtype = int)
# tracks errors
error_count = 0;

# Follows tree and compares example label with tree outcome
for i in range(num_exp) :
   tree_outcome_test[i] = branches_tree[0,allterms_test[i,attr_tree[0]]]
   while tree_outcome_test[i] >= 0:
        tree_outcome_test[i] = branches_tree[tree_outcome_test[i],allterms_test[i,attr_tree[tree_outcome_test[i]]]]
   tree_outcome_test[i] = -1*(tree_outcome_test[i]+1)
   if tree_outcome_test[i] != labels_test[i]:
        error_count = error_count + 1
        
test_error = float(error_count)/float(num_exp)
print('Test Error:', test_error)