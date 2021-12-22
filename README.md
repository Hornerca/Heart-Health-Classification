# Heart Heath Classification- Model Selection, Visulization in Sklearn (RandomForrests, Bagging, SVM, ExtraTree, kNN)

According to the Canadian Chronic Disease Surveillance System, 1 in 12 individuals over 20 years old are living with a heart disease in 2012-2013. Using data science, this repository explores model selection based on cross validated accuracy, data visulization and feature importance.
Database: https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive#files

- Each model's hyper-parameters tuned using grid search
- 10-fold corss validation repleated 10 times was performed on each model 
- Model selection based on averaged accuracy
- ROC Curve Presented
- Permutation Importance of Each Feature

## Classifiers Accuracies- 10 fold cross validation, repeated 5 times

Accuracy | Classifier
------------ | -------------
0.9172 | ExtraTree
0.9168 | RandomForests
0.8872 | Bagging
0.8502 | kNN
0.8498 | SVM


## Selected Classifier: Extra Tree 
info   | precision   | recall | f1-score  | support
------- | ----------- |-------------- | ---------- | ----------
0   |    0.89  |    0.75   |   0.82   |    112
1   |    0.81   |   0.92    |  0.86   |    126
accuracy   |           |            | 0.84    |   238
macro avg    |   0.85    |  0.84   |   0.84    |   238
weighted avg    |   0.85    |  0.84   |   0.84   |    238


## ROC Curve
An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. 
This curve plots two parameters:
- True Positive Rate
- False Positive Rate

![Alt Text](https://github.com/Hornerca/Heart-Health-Classification/blob/main/ROC.png)

## Non-parametric Feature Importance
This data suggest suggests that there are 6 main feature that are different between classes. Less features prevent overfitting and minimize computation costs. Feature importance exploration also alows for meaningfull insight into the data. 
![Alt Text](https://github.com/Hornerca/Heart-Health-Classification/blob/main/Permutation%20Feature%20Importance.png)
