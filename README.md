# Heart Heath Classification- Model Selection, Visualization in Sklearn (RandomForrests, Bagging, SVM, ExtraTree, kNN)

According to the Canadian Chronic Disease Surveillance System, 1 in 12 individuals over 20 years old are living with a heart disease in 2012-2013. Using data science, this repository explores model selection based on cross validated accuracy, data visualization and feature importance.

Database: https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive#files

- Each model's hyper-parameters tuned using grid search
- 10-fold cross validation repeated 10 times was performed on each model 
- Model selection based on cross validated accuracy
- ROC Curve of Selected Model
- Permutation Importance of Each Feature

## Classifiers Accuracies- 10 fold cross validation, repeated 5 times

Accuracy | Classifier
------------ | -------------
0.9172 | ExtraTree
0.9168 | RandomForests
0.8872 | Bagging
0.8502 | kNN
0.8498 | SVM


## Selected Classifier Summary on Testing Set: Extra Tree 
info | precision   | recall | f1-score  | support
------- | ----------- |-------------- | ---------- | ----------
Healthy (0)   |    0.89  |    0.75   |   0.82   |    112
Heart Disease (1)   |    0.81   |   0.92    |  0.86   |    126
accuracy   |           |            | 0.84    |   238
macro avg    |   0.85    |  0.84   |   0.84    |   238
weighted avg    |   0.85    |  0.84   |   0.84   |    238


## ROC Curve for Extra Tree
An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. 
This curve plots two parameters:
- True Positive Rate
- False Positive Rate

![Alt Text](https://github.com/Hornerca/Heart-Health-Classification/blob/main/ROC.png)

## Non-parametric Feature Importance
This data suggests there are 6 main features that are different between classes, with exercise angina being the best. Advantages of using less features in a model, prevents overfitting and minimize computation costs. Feature importance provides meaningful insight into the data as well. 

![Alt Text](https://github.com/Hornerca/Heart-Health-Classification/blob/main/Permutation%20Feature%20Importance.png)
