---
layout: post
title:      "TELECOM CUSTOMER ATTRITION ANALYSIS"
date:       2020-07-14 15:02:03 +0000
permalink:  telecom_customer_attrition_analysis
---


     Customer attrition or also known as customer churn is the number of paying customers who fail to become repeat customers. In other words, it is the loss of customers by a company. There are two types of customer attrition: involuntary and voluntary. Involuntarily churn is the stop in a customer-business relationship that is made by the company. It is usually due to the customers' financial or logistical responsibilities. The second type, voluntarily churn is the customers' decision to make a stop in purchasing the company's product or service. This loss is often caused by the customers' perception that the company's products are no longer align with the customers' needs and/or values. According to a study by Bain & Company and the Harvard Business School, decreasing customer attrition rate by 5% will increase the company profits by 25% to 95%. On the other hand, although both new and existing customer groups cannot be neglected, numerous articles have demonstrated that acquiring new customers is more expensive than retaining existing customers. Therefore, analysis of customer attrition is absolutely important to help a company understand the causes of customer churn and gain new insight into its strategic planning for future improvement.
 
     For this topic, I chose the dataset that was gathered from SyriaTel Company and was uploaded on Kaggle. Originally, the dataset consists of more than 3,300 customers. About 86% of these entries are non-churned customers and 14% are churned customers. In addition, a target of 'churn' and 20 features are included in the dataset which I placed into different categories:
		 
· Customer Information: state, area code, phone number
· Service Plans: international plan, voice mail plan
· Service Charge: number of voicemail messages, total day minutes/ calls/ charge, total eve minutes/ calls/ charge, total night minutes/ calls/ charge, total intl minutes/ calls/ charge, customer service calls

[https://www.kaggle.com/becksddf/churn-in-telecoms-dataset](http://)

     Most of the datasets consist of uncleaned data. As you may know, uncleaned data can create misleading in terms of understanding the real causes of customer churn through this analysis. With that being said, the first crucial step is cleaning the dataset. Any missing values, duplicate data points, and outliers are properly taken care of. After that, any columns that are possibly not a contributor for building models later were also eliminated such as 'area code' and 'phone number' columns. In addition, a new feature column was created by adding 'total day charge', 'total night charge', 'total eve charge', 'total intl charge'. It is called 'total charge' which is the total amount that customers have to pay monthly. The cleaned data which now consists of about 3,200 entries was saved in a new csv file for data analysis later.

     The categorical feature columns: state, international plan, voice mail plan in this dataset were originally in yes/no form. Therefore, they are label-encoded into numeric values: 1 for yes and 0 for no. With continuous feature columns, I scaled these variables by using StandardScaler()
		 
		 `label_encode = LabelEncoder()

df['state'] = label_encode.fit_transform(df['state'])
df['international plan'] = label_encode.fit_transform(df['international plan'])
df['voice mail plan'] = label_encode.fit_transform(df['voice mail plan'])


ss = StandardScaler()
scaled = ss.fit_transform(df[con_feats])
scaled = pd.DataFrame(scaled, columns=con_feats)`

     I initially selected the features by using Ordinary Least Squares (OLS) based on p-values and LassoCV based on coefficient with of course the multicollinearity check beforehand. According to OLS, the important features are international plan, total intl calls, customer service calls, and voice mail plan. Similarly, LassoCV claimed that international plan, total charge, customer service calls, voice mail plan, and total intl calls are the important features. 
 
      For this project, I used five different classifiers to build my models: Logistic Regression, Random Forest, K-Nearest Neighbor, Decision Tree, and XGBoost. Each of the models was run through hyperparameter grid search by GridSearchCV with defined parameter grids and StratifiedKFold of 5 splits.
			
			`def gridsearch_cv(clf, params, X_train, y_train, cv):
    pipeline = Pipeline([('clf', clf)]) 
    gs = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring='f1', return_train_score=True)
    gs.fit(X_train, y_train)
    return gs`
		
		`logistic_regression_params = {'clf__solver': ['liblinear'], 
                              'clf__C': [0.1, 1, 10],
                              'clf__penalty': ['l2', 'l1']}

random_forest_params = {'clf__max_depth': [25, 50, 75],
                        # just sqrt is used because values of log2 and sqrt are very similar for our number of features (10-19)
                        'clf__max_features': ['sqrt'],
                        'clf__criterion': ['gini', 'entropy'],
                        'clf__n_estimators': [100, 300, 500, 1000]}

knn_params = {'clf__n_neighbors': [5, 15, 25, 35, 45, 55, 65],
              'clf__weights': ['uniform', 'distance'],
              'clf__p': [1, 2, 10]}

decision_tree_params = {'clf__max_depth': [25, 50, 75],
                        'clf__max_features': ['sqrt'],
                        'clf__criterion': ['gini', 'entropy'],
                        'clf__min_samples_split': [6, 10, 14],}

xgb_params = {'clf__learning_rate': [0.1],
              'clf__max_depth': [1, 3, 6],
              'clf__n_estimators' : [30, 100, 250],
              'clf__min_child_weight': [4, 6, 10],
              'clf__subsample': [0.7]}`
							
     After running all five of the classifiers and conducting their confusion matrices, the  Receiver Operating Characteristic (ROC) curve was plotted to see how well each model distinguishes between classes. Higher the AUC, the better the model is at predicting 0s as 0s and 1s as 1s. By analogy, the higher the AUC, the better the model is at distinguishing between churned customers and non-churned customers. XGBoost has the highest AUC value, followed by RandomForest, DecisionTree, LogisticRegression. K-Nearest Neighbor has the lowest AUC value.
		 
		 
		 After visualizing how well each model performs, another way to choose the best performing model is by calculating metrics on test sets of each classifier. Below are the four common metrics that are calculated for each classifier 
* Accuracy: How often is the classifier correct?
Can be misleading with unbalanced datasets

* Precision: When it predicts yes, how often is it correct?
Can be misleading if the model is very conservative about which cases they predict as true

* Recall: When it is actually yes, how often does it predict yes?
Can get a 100% recall score by simply assuming every case is true, which would not necessarily be a better model

* F1: The harmonic average of precision and recall.
F1 cannot be high without both precision and recall being high. Therefore, in this case, F1 score is a great metric to choose which model is the best along with AUC.

XGBoost has the highest AUC value and also the highest F1 score. RandomForest has the second-highest AUC value and F1 score. However, it took a long time to run, compared to other classifiers. It is not a good model as it takes more time and memory storage. DecisionTree comes in third place and Logistic Regression comes in fourth place. K-Nearest Neighbor has the lowest AUC value and also the lowest F1 score. To sum, XGBoost is the model that has the best performance.

### THE FINAL MODEL 
XGBoost was chosen to be the best model out of five built models. After that, the feature importance is obtained to identify which features affect the customer attrition rate for SyriaTel company. According to XGBoost classifier, the five most important factors that have great impacts on churned customers are 'total charge', 'voicemail plan','total international calls', 'customer service calls', 'international plan'. Some of these features overlap with features that are initially selected from both OLS and LassoCV: 'international plan' and 'customer service calls'. To reduce the customer attrition rate, SyriaTel should focus on those factors above.

