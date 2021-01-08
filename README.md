# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset used in this project contains information about a Bank's Marketing Data. It has 21 columns and about 33000 rows. The aim is to predict if the client will subscribe to a fixed term deposit(y). This is one of the sample datasets provided by Azure and the dataset can be found at- https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

As already stated above, a comparison is done between the performance of the Logistic Regression model and Azure AutoML run. It was observed that the accuracy with LightGBM and Logistic Regression model was 0.9149 and 0.9146 respectively. 

## Scikit-learn Pipeline
1. Loading and cleaning data: First a tabular dataset was loaded from TabularDatasetFactory to do some exploration and understand the meaning of each feature, once done data was prepared and cleaned by using one-hot encoding technique to deal with discrete features, after that, data was split into training and testing sets.

2. Training- Next step was to use Logistic Regression for the prediction task which in this case is classification. It was accompanied this with Hyperparameter Tuning using Azure's Hyperdrive to tune 'C' and 'max_iter' hyperparameters. RandomParamterSampling was used to choose hyperparamters and it randomly chooses these from a defined search space which could contain discrete or continuous values depending upon the requirement. The evaluation metric chosen to be maximised was Accuracy. Then the Termination Policy was defined for every run using BanditPolicy based on a slack factor equal to 0.1 as criteria for evaluation to conserve resources by terminating runs that are poorly performing. Next, an SKLearn estimator was created, the hyperdrive configuration was defined, and finally, the hyperparameter tuning job was launched.

3. Results- Finally, this pipeline was executed multiple times with some modifications to the Hyperdrive configuration to improve the Accuracy. In this case the best model was generated using this hyperparameters (C = '1', max_iter = '200') and gave an Accuracy of 0.9146

## AutoML
1. Data Preparation- The same steps followed in Scikit Learn pipeline were adopted here. 
2. Configure AutoML Run- Here the type of the task: Classification, the primary metric: Accuracy, the data, the column to be predicted and the constraints were specified. Finally Call the submit method on the experiment object and pass the run configuration. Once finished, register the model for future use. In this case, the best model was generated using LightGBM Algorithm which gave an Accuracy of 0.9149 

## Pipeline comparison
To analyze the distinction among the two models Accuracy as a primary metric was used and the outcome was that Auto ML provides more high-grade performance. This result is coherent mostly because Auto ML run not only test more hyperparameter value than the Scikit-learn process but also more algorithms too.

## Future work
The improvement can be made not only in the Auto ML process by not using the cleaned data function (train.py) and leave the featurization to the Auto ML run (to handle the Imbalanced data), but also in the Scikit-Learn process by using other algorithm and testing other configuration to tune the hyperparameters.

## Proof of cluster clean up
![image](https://github.com/gunisha30/Optimizing_MLPipeline_Azure/blob/main/deleteinstance.png)
