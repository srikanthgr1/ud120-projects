# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 00:58:22 2018

@author: M1033493
"""


def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
 
    
    ### your code goes here!
    
    from sklearn.naive_bayes import GaussianNB  
    from sklearn import svm
#    opt = GaussianNB()
#    opt.fit(features_train,labels_train )
    opt = svm.SVC(C=10000.0,  kernel='rbf', gamma=1000.0)
    opt.fit(features_train, labels_train)  
     
    return opt