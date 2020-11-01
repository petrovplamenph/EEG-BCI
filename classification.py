# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:43:19 2019

@author: EMO
"""
import numpy  as np

from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

import pandas as pd
def eval_rbf_svm(X_train, X_test, y_train, y_test):
    """
    Hyperparametar tune and optimize SVM classifier with RBF kernel
    Get the accuracy of the classifier on a test set 
    """
    parameter_candidates = [
      {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma':   [0.12, 0.1,0.08, 0.01], 'kernel': ['rbf']},
    ]
    RBF_SVM = GridSearchCV(estimator=SVC(), cv=3 ,param_grid=parameter_candidates)

    RBF_SVM.fit(X_train, y_train) 
    y_pred = RBF_SVM.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(RBF_SVM.best_estimator_)
    print(acc)
    return acc


def eval_sigmoid_svm(X_train, X_test, y_train, y_test):
    """
    Hyperparametar tune and optimize SVM classifier with RBF kernel
    Get the accuracy of the classifier on a test set 
    """
    parameter_candidates = [
      {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['sigmoid']},
    ]
    sig_SVM = GridSearchCV(estimator=SVC(), cv=3 ,param_grid=parameter_candidates)

    sig_SVM.fit(X_train, y_train) 
    y_pred = sig_SVM.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(sig_SVM.best_estimator_)
    print(acc)
    return acc

def eval_linier_svm(X_train, X_test, y_train, y_test):
    """
    Hyperparametar tune and optimize Liniear SVM classifier 
    Get the accuracy of the classifier on a test set 
    """
    parameter_candidates = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

    LIN_SVM = GridSearchCV(estimator=LinearSVC(), cv=3 ,param_grid=parameter_candidates)

    LIN_SVM.fit(X_train, y_train) 
    y_pred = LIN_SVM.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(LIN_SVM.best_estimator_)
    print(acc)
    return acc





def eval_logit_regression(X_train, X_test, y_train, y_test):
    """
    Hyperparametar tune and optimize Logistic Regression classifier 
    Get the accuracy of the classifier on a test set 
    """
    parameter_candidates = [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

    LogitR = GridSearchCV(estimator=LogisticRegression(), cv=3 ,param_grid=parameter_candidates)

    LogitR.fit(X_train, y_train) 
    y_pred = LogitR.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(LogitR.best_estimator_)
    print(acc)
    return acc

def eval_knn(X_train, X_test, y_train, y_test):
    """
    Get the accuracy of KNN classifier(n_neighbours=3) on a test set 
    """
    #parameter_candidates = [{'n_neighbors': [2, 3, 5, 7]}]

    #knn = GridSearchCV(estimator=KNeighborsClassifier(), cv=3 ,param_grid=parameter_candidates)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train) 
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    return acc

def eval_MultinomialNB(X_train, X_test, y_train, y_test):
    """
    Get the accuracy of  MultinomialNB classifieron a test set 
    """
    NB = MultinomialNB()
    NB.fit(X_train, y_train) 
    y_pred = NB.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    return acc


def eval_DT(X_train, X_test, y_train, y_test):
    """
    Hyperparametar tune and optimize Decision Tree classifier 
    Get the accuracy of the classifier on a test set 
    """
    parameter_candidates = [{'criterion': ['gini', 'entropy'],
                             'min_samples_split':[2,3],
                             'min_samples_leaf':[1,2]}]

    DT = GridSearchCV(estimator=DecisionTreeClassifier(), cv=3 ,param_grid=parameter_candidates)

    DT.fit(X_train, y_train) 
    y_pred = DT.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(DT.best_estimator_)
    print(acc)
    return acc

def train_test(X_train, X_test, y_train, y_test):
    """
    On a given train dataset train variety of classifiers and evaluate them o the test set
    finaly return the results 
    """
    
    X_train, y_train = shuffle(X_train, y_train)

        
    y_train = [int(label) for label in y_train]
    y_test = [int(label) for label in y_test]
            
        
    print('RBF SVM ')
    rbf_acc = eval_rbf_svm(X_train, X_test, y_train, y_test)
        
    print('linear SVM ')
    lin_svm_acc= eval_linier_svm(X_train, X_test, y_train, y_test)
        
    print('KNN Euclidian')
    knn_acc = eval_knn(X_train, X_test, y_train, y_test)
            

    print('DT')
    dt_acc = eval_DT(X_train, X_test, y_train, y_test)
    report = [rbf_acc, lin_svm_acc, knn_acc, dt_acc]
    algorithams_names = ['RBF SVM', 'linear SVM', 'KNN Euclidian', 'Decision tree'] 
    return report, algorithams_names
    
