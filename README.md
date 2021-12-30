# Heart Heath Classification- Model Selection, Feature Importance, Visualization in Sklearn (RandomForrests, Bagging, SVM, ExtraTree, kNN)

According to the Canadian Chronic Disease Surveillance System, 1 in 12 individuals over 20 years old are living with a heart disease in 2012-2013. Using data science, this repository explores model selection based on cross validated accuracy, data visualization and feature importance.

Database: https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive#files

- Each model's hyper-parameters tuned using grid search
- 10-fold cross validation repeated 10 times was performed on each model 
- Model selection based on cross validated accuracy
- Statistical test between classifier's cross validated accuracies
- ROC Curve of Selected Model
- Permutation importance of each feature

## Classifiers Accuracies- 10 fold cross validation, repeated 5 times

Accuracy | STD | Classifier
------------ | -------------| ---
0.9658| 0.0155| ExtraTree
0.9551| 0.0207| RandomForests
0.9394| 0.0235| Bagging
0.9092| 0.0248| SVM
0.8979| 0.0315| kNN

![Alt Text](https://github.com/Hornerca/Heart-Health-Classification/blob/main/Accuracy%20Box%20Plots.png)

Figure: Boxplot of cross validated accuracies, all above chance level. 

- The below figure illustrates that there is no significant difference between the classifier with the highest accuracy (ExtraTree) and RandomForests. All other classifiers show significant difference between classifiers.  

![Alt Text](https://github.com/Hornerca/Heart-Health-Classification/blob/main/Dunn_Pairwise_Classifier_Comparison.png)

Figure: Dunn Pairwise comparison of 4 classifiers using Bonferroni adjustment. P-values outlined in legend with NS representing no significance. 


## Selected Classifier Summary on Testing Set: Extra Tree 
info | precision   | recall | f1-score  | support
------- | ----------- |-------------- | ---------- | ----------
Healthy (0)     |    0.93    |    0.79    |    0.85    |     114
Heart Disease (1)    |     0.81   |     0.94    |    0.87    |     111
accuracy         |              |         | 0.86     |    225
macro avg       |  0.87      |  0.86     |   0.86      |   225
weighted avg      |   0.87     |   0.86    |    0.86     |    225


## ROC Curve for Extra Tree
An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. 
This curve plots two parameters:
- True Positive Rate (Correct classification of Heart Disease diagnosis)
- False Positive Rate ( Incorrect classification of Heart Disease diagnosis)

![Alt Text](https://github.com/Hornerca/Heart-Health-Classification/blob/main/ROC.png)

## Non-parametric Feature Importance
This data suggests there are 6 main features that are different between classes, with exercise angina being the best. Advantages of using less features in a model, prevents overfitting and minimize computation costs. Feature importance provides meaningful insight into the data as well. 

![Alt Text](https://github.com/Hornerca/Heart-Health-Classification/blob/main/Permutation%20Feature%20Importance.png)
