# Machine-learning
The repository contains some of the most used ML python algorithms along with implementation in R for some of them. 

Python packages used : numpy,pandas,matploit,sklearn,statsmodels,keras,nltk,........continues will be added more.

# Regression
1) In linear regression, dataset contains details of employee salary and years of experience and using this model we can predict the salary of employee by years of experince.

2) In multiple linear regression, dataset contains details of expenditure of startups and their profit and using this model we can predict the profit of startup, also backward elimination technique is implemented for the same.

3) In polynomial linear regression, dataset contains details of salary and years of experience same as descibed earlier. Such analysis could help HR dept. in salary negotiations during candidate hiring process.

4) In SVR linear regression, Decision Tree Regression and Random Forest Regression we have used the salary and experience dataset again with different algorithmic approach for prediction. Random Forest algorithm outperforms others.

# Classification
1) In logistic regression, dataset containing details of salary, age and product purchase and using this model we can predict the whether the customer of certain age and salary will buy the product or not .

2) We have used same dataset for classification using KNN, SVM, Kernel SVM, Baive Bayes, SVM, Decision Tree and Random Forest to compare the performance of different algorithms.  


# Clustering 

1) In k-means clustering is example of segmentation task and we have used a dataset containing details of gender, age, score etc to segment customers into clusters.

2) In Hierarchical Clustering we have used dataset that will help us to cluster people according to their income & spending habits in malls.

Hierarchical Clustering uses concept of dendograms to identify most appropriate number of clusters for segmentation.

# Association Rule Learning

1)  Market Basket Analysis is a machine learning-based technique for identifying buying pattern from numerous retail transactions and helping the retailer in increasing the sales. Apriori Algorithm was implemented to find relatiobship between purchasing behaviour of customers. 

2) Analysing Market basket using Eclat algorithm (similar to Apriori) also tries to club products which are purchased together.

# Dimensionality Reduction
In statistics, machine learning, and information theory, dimensionality reduction is the process of reducing the number of predictors under consideration by obtaining a set of principal variables to act as new predictors for target value.

Following algorithms are implemented for dimensionality reduction: 
1) PCA algorithm 
2) Kernel_pca 
3) LDA 

# Deep Learning
1) Artificial Neural Network 

ANN model for understanding churn at a bank by taking into account several factors which affect the retention of customers

2) Convolutional Neural Network

Convolutional neural networks have been the most influential innovations in the field of computer vision. CNN model in python as a dog or cat identifier.

# Natural Language Processing
Natural Language Processing (or NLP) is applying Machine Learning models to text and language. Teaching machines to understand what is said in spoken and written word is the focus of Natural Language Processing. Code implements the following aspects of NLP:

+ Clean texts to prepare them for the Machine Learning models
+ Create a Bag of Words model
+ Apply Machine Learning models onto this Bag of Worlds model

# Model Selection 

1) XGBoost

XGboost is a very fast, scalable implementation of gradient boosting that has taken data science by storm, with models using XGBoost regularly winning many online data science competitions and used at scale across different industries
 
 
2) K cross-fold validation

Cross-validation is a procedure used to evaluate machine learning models for long run accuracy metrics.

The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=10 becoming 10-fold cross-validation.

3) Grid CV search

GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.

The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

# Reinforcement Learning

Reinforcement Learning is a branch of Machine Learning, also called Online Learning. It is used to solve interacting problems where the data observed up to time t is considered to decide which action to take at time t + 1. It is also used for Artificial Intelligence when training machines to perform tasks such as walking. Desired outcomes provide the AI with reward, undesired with penalty. Machines learn through trial and error. Algorithms implemented:

1. Upper Confidence Bound (UCB)
2. Thompson Sampling





