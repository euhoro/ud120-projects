#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

clf1 = DecisionTreeClassifier(min_samples_split=40)
rand1 = RandomForestClassifier()
knear = KNeighborsClassifier()
adaboost = AdaBoostClassifier()


def train_and_classify(clf):
    t0 = time()
    # < your clf.fit() line of code >
    # len(features_train[0]
    clf.fit(features_train, labels_train)
    print("Training Time:", round(time() - t0, 3), "s")
    t0 = time()
    # < your clf.fit() line of code >
    results = clf.predict(features_test)
    print("Predicting Time:", round(time() - t0, 3), "s")
    print("Accuracy:", clf.score(features_test, labels_test))
    print("num of features:", len(features_train[0]))


train_and_classify(adaboost)

#########################################################


