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
from scipy import stats
import scikit_posthocs as sp #pip install scikit-posthocs

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

y_csv=np.array(df['target'])
X_csv=np.array(df.drop(['target'],axis=1))

#%% ensure class sizes are equal to prevent classification bias
unique, counts = np.unique(y_csv, return_counts=True)
print ('unbalance classes',np.asarray((unique, counts)).T) # class size not equal

'''
Danger of imbalanced datasets are that the model will be inherantly biased towards
the class with more samples. 
 
Ways to takle uneven datasets are upsampling minority class and downsampling
the majority class. In this example we are downsampling classes to match
'''

#split data sets
X_normal=X_csv[y_csv==0]
y_normal=y_csv[y_csv==0]

X_disease=X_csv[y_csv==1]
y_disease=y_csv[y_csv==1][:len(y_normal)]

# downsample eyes_closed data set
X_disease_ds=resample(X_disease, 
                        replace=False,    # sample without replacement
                        n_samples=len(y_normal),     # to match minority class
                        random_state=123) # reproducible results

X=np.vstack([X_normal,X_disease_ds])
y=list(y_normal) +list(y_normal+1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# scale features
min_max_scaler = preprocessing.MinMaxScaler()
X_featScaled = min_max_scaler.fit_transform(X_train)

#shuffle data
X_train,y_train=shuffle(X_featScaled,y_train, random_state=273377)

#%% Tune Hyper parameters & train models
models_tune={}
score='roc_auc'
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
                    'Bagging': BaggingClassifier(random_state=1).set_params(**models_tune['Bagging'].best_params_),
 
                    'ExtraTree': ExtraTreesClassifier(random_state=1).set_params(**models_tune['ExtraTree'].best_params_),
                    
                    'kNN':KNeighborsClassifier().set_params(**models_tune['kNN'].best_params_),
                    
                    'SVM':SVC(random_state=1).set_params(**models_tune['SVM'].best_params_),                                 
                    
                    'RandomForests': RandomForestClassifier(random_state=1).set_params(**models_tune['RandomForests'].best_params_),

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

df_scores=pd.DataFrame(scores, index=keys)
bplot=df_scores.T.boxplot(grid=False, rot=0)
bplot.set_ylabel('Accuracy')
bplot.set_title('10-fold Cross Validation Accuracy')
bplot.figure.savefig('Permutation Feature Importance.png')



print('Classifiers- 10 fold cross validation- repeat 5x')
for i in results:
    i=list(i)
    print(np.round(i[0],4), i[1])
     
print(' ')

print('best model to use:',results[0][1])

#%% statistically best classifier

# normality={}
# kw={}
# kw_all={}
# dunn={}
# p_value={}
heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}

#classifier Selection for First online Session
fig, ax = plt.subplots()
   
# normality- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
temp_data=df_scores.T
p_all=[]
for key in temp_data:
    stat,p=stats.shapiro( temp_data[key])
    p_all.append(p)

#Kruskal-Wallis test - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
samples = [condition for condition in temp_data.T.values.tolist()]

#%post hoc- https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn/
dunn=sp.posthoc_dunn(temp_data.T.values.tolist(),p_adjust = 'bonferroni').set_index(temp_data.columns).T.set_index(temp_data.columns)
t=sp.sign_plot(dunn,ax=ax, **heatmap_args)


title="Dunn Pairwise Classifier Comparison "
fig.suptitle(title)  
fig.tight_layout()
fig.savefig('Dunn_Pairwise_Classifier_Comparison.png')


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
disp.figure_.savefig('ROC.png')

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
plt.savefig('Permutation Feature Importance.png')









