#!/usr/bin/python
import sys
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


#### your code goes here


from sklearn import tree
from sklearn.metrics import accuracy_score
# clf = tree.DecisionTreeClassifier()

clf1 = tree.DecisionTreeClassifier(min_samples_split = 2)
clf1 = clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)
acc_min_samples_split_2 = accuracy_score(labels_test,pred1 )


clf2 = tree.DecisionTreeClassifier(min_samples_split = 50)
clf2 = clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)
acc_min_samples_split_50 = accuracy_score(labels_test,pred2 )




try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
