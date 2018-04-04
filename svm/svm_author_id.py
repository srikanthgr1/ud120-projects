#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB  
from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###


'''
kernel : string, optional (default=’rbf’)
Specifies the kernel type to be used in t.he algorithm. 
It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. 
If none is given, ‘rbf’ will be used. 
If a callable is given it is used to pre-compute the kernel matrix from data matrices; 
that matrix should be an array of shape (n_samples, n_samples)

gamma : float, optional (default=’auto’)
Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
If gamma is ‘auto’ then 1/n_features will be used instead.
'''

#features_train = features_train[:round(len(features_train)/100)] 
#labels_train = labels_train[:round(len(labels_train)/100)]

t0 = time()
knl = "rbf" 
cval = 10000.0
# Change the kernel value to others and also play with the C, gamma values in SVC method
opt = svm.SVC(C= cval,kernel=knl)
opt.fit(features_train, labels_train)
print("SVM training Time with Kernel as " ,knl.title() ," and C value as", cval,"is", round(time()-t0, 3))

t1 = time()
pred_features_test = opt.predict(features_test)
print("SVM Prediction Time with Kernel as " ,knl.title() ," and C value as", cval,"is",  round(time()-t1, 3))
print("The Accuracy of SVM on Predicting Authors with Kernel as" ,knl.title() ,"and C value as", cval,"is"   , opt.score(features_test,labels_test))


#########################################################

