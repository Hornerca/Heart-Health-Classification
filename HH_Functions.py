#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 20:22:07 2021

@author: christine.horner
"""


import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd

from sklearn.utils import resample
from scipy.integrate import simps
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from operator import itemgetter

from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold,cross_val_score
    
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5,random_state=223)



def tune_hyper_parms(model,X,y,score):

  
    #%tuning hyper parameters-------------------------------
    '''
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
    
    '''
    if model=='kNN':
 
        #% kNN
        # Set the parameters by cross-validation
        '''
        https://medium.com/@mohtedibf/in-depth-parameter-tuning-for-knn-4c0de485baf6
        '''
        tuned_parameters = {'n_neighbors':list(range(3,11,1)),
                            # 'weights':['uniform','distance'],
                            # 'algorithm':['auto','ball_tree','kd_tree','brute'],
                            # 'p':[1,2]
                            }
                            
        clf = GridSearchCV(
                KNeighborsClassifier(), tuned_parameters, 
                scoring= score,cv=rskf
                )
        return clf.fit(X, y)
  
    elif model=='SVM':    
        tuned_parameters = {
                            'C':list(np.arange(0.1,1,0.1)),
                            'degree':[1,2,3]
           
                            }
        
        clf = GridSearchCV(
            SVC(), tuned_parameters, 
            scoring= score,cv=rskf
        )
        return clf.fit(X, y)
        
    elif model=='Bagging':

        #% Bagging
        
        tuned_parameters={ 'base_estimator':[SVC(),DecisionTreeClassifier() ],
                          # 'n_estimators':list(range(10,150,10)),
                           # 'bootstrap':[True]
                     
            }
        
        clf = GridSearchCV(
            BaggingClassifier(), tuned_parameters, 
            scoring= score,cv=rskf
        )
        return clf.fit(X, y)
    
    elif model==  'RandomForests' :
        #% Random Forrests
        
        '''
        https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        
        
        '''
        
        tuned_parameters={ 'n_estimators':list(range(50,150,25)),
                          'max_features':['auto', 'sqrt', 'log2'], #list(range(5,13,2)),
                          # 'min_samples_split': list(range(2,6)),
                          # 'max_depth':list(range(8,13)),
                          
                          'criterion':['gini','entropy'],
          
            }
        
        clf = GridSearchCV(
            RandomForestClassifier(), tuned_parameters, 
            scoring= score,n_jobs=-1,cv=rskf
        )
        return clf.fit(X, y)
   
    
    elif model==  'ExtraTree':  
        #% ExtraTree- max_depth, n_estimators, 
        '''
        https://machinelearningmastery.com/extra-trees-ensemble-with-python/
        https://machinelearningmastery.com/overfitting-machine-learning-models/
        '''
        
        tuned_parameters={ 'n_estimators':list(range(50,150,25)),
                          'max_features': ['auto', 'sqrt', 'log2'],#list(range(5,13,2)),
                          # 'min_samples_split': list(range(2,6)),
                          # 'max_depth':list(range(8,13)),
                          
                          'criterion':['gini','entropy'],
                          }
        
        clf = GridSearchCV(
            ExtraTreesClassifier(), tuned_parameters, 
            scoring= score,n_jobs=-1,cv=5
            )
        return clf.fit(X, y)
   
    else:
        print('model not found',model)