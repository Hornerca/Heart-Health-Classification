#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 19:29:28 2021

@author: christine.horner
"""
#Heart Health
from HH_Functions import *

import numpy as np
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd

from sklearn.utils import resample
from scipy.integrate import simps
from sklearn import preprocessing,metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from operator import itemgetter
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold,cross_val_score
    
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle


#read CSV file (
# https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive#files
df=pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

y=np.array(df['target'])
X=np.array(df.drop(['target'],axis=1))

#%% ensure class sizes are equal to prevent classification bias
unique, counts = np.unique(y, return_counts=True)
print ('unbalance classes',np.asarray((unique, counts)).T) # class size not equal

'''
Danger of imbalanced datasets are that the model will be inherantly biased towards
the class with more samples. 
 
Ways to takle uneven datasets are upsampling minority class and downsampling
the majority class. In this example we are downsampling classes to match
'''

#split data sets
X_normal=X[y==0]
y_normal=y[y==0]

X_disease=X[y==1]
y_disease=y[y==1][:len(y_normal)]

# downsample eyes_closed data set
X_disease_ds=resample(X_disease, 
                        replace=False,    # sample without replacement
                        n_samples=len(y_normal),     # to match minority class
                        random_state=123) # reproducible results


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scale features
min_max_scaler = preprocessing.MinMaxScaler()
X_featScaled = min_max_scaler.fit_transform(X_train)

#shuffle data
X_train,y_train=shuffle(X_featScaled,y_train, random_state=273377)

#%% Tune Hyper parameters & train models
models_tune={}
score='accuracy'
#% kNN    
models_tune['kNN']=tune_hyper_parms('kNN',X_train,y_train, score)

#% LDA   
models_tune['SVM']=tune_hyper_parms('SVM',X_train,y_train,score)

#% Bagging
models_tune['Bagging']=tune_hyper_parms('Bagging',X_train,y_train,score)

#% Random Forrests  
models_tune['RandomForests']=tune_hyper_parms('RandomForests',X_train,y_train,score)
   

#% ExtraTree    
models_tune['ExtraTree']=tune_hyper_parms('ExtraTree',X_train,y_train,score) 

# train classifiers using optimal hyperparameters
classifiers = {    
                    'Bagging': BaggingClassifier().set_params(**models_tune['Bagging'].best_params_),
 
                    'ExtraTree': ExtraTreesClassifier().set_params(**models_tune['ExtraTree'].best_params_),
                    
                    'kNN':KNeighborsClassifier().set_params(**models_tune['kNN'].best_params_),
                    
                    'SVM':SVC().set_params(**models_tune['SVM'].best_params_),                                 
                    
                    'RandomForests': RandomForestClassifier().set_params(**models_tune['RandomForests'].best_params_),

            }  


#%% Model Validation

#set up cross validation
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5,random_state=36851234)

keys=list(classifiers.keys())
params=[str(models_tune[key].best_params_) for key in keys]
scores=[[] for x in keys]

for idx,key in enumerate(keys):
    model=classifiers[key]
    scores[idx] = cross_val_score(model, X_train, y_train, scoring=score, cv=rskf, n_jobs=-1, error_score='raise')
   
    
results=sorted(list(zip(np.mean(scores,axis=1),keys,np.std(scores,axis=1),params, scores) ),reverse=True)
    
print('Classifiers- 10 fold cross validation- repeat 5x')
for i in results:
    i=list(i)
    print(np.round(i[0],4), i[1])
     
print(' ')

print('best model to use:',results[0][1])


#%% Performance Results

#scale featues based on min_max_scaler function
X_featScaled_test = min_max_scaler.fit_transform(X_test)

#using the model with the highest accuracy (RandomForests)
clf=classifiers[results[0][1]]
clf.fit(X_train,y_train)
y_predict=clf.predict(X_featScaled_test)

#classification report
print(classification_report(y_test,y_predict))

#ROC of Best Classifier
disp=metrics.plot_roc_curve(clf, X_test, y_test) 
disp.figure_.suptitle("ROC curve comparison")

#%%Permutation-based feature importance
result = permutation_importance(clf, X, y, scoring=score,n_repeats=100, random_state=0, n_jobs=-1)

feature_names=df.T.index

fig, ax = plt.subplots()
sorted_idx = result.importances_mean.argsort()
ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=feature_names[sorted_idx]
)
ax.set_title("Permutation Importance of each feature")
ax.set_ylabel("Features")
fig.tight_layout()
plt.show()










