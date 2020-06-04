---
layout: post
title:      "KING COUNTY HOUSE SALE MARKET"
date:       2020-06-04 20:33:18 +0000
permalink:  king_county_house_sale_market
---


![](https://github.com/linhmai19/dsc-mod-2-project-v2-1-online-ds-sp-000/blob/master/seattle.jpg)


## INTRODUCTION
Doubtlessly, this century is the century of advanced technology. Beyond the famous Silicon Valley of California, Seattle has been growing to be one of the biggest tech hubs in the U.S. This fast growing of Seattle leads to an increase in population and eventually leads to a higher demand in housing. A price of a house is affected by many factors such as locations, the footage of the house, the condition, and so on. On the other hand, as a real estate company, people also needs to see whether they should invest in expensive areas with less houses sold or invest in more affordable areas with more houses sold. Therefore, in order for the real estate company to have a better understanding about the home sale market in Seattle or King County in general make a decision on how to properly invest in this market, a data analysis is done on King County home sales data between the period of May 2014 and May 2015.

## DATASET OVERVIEW AND DATA CLEANING
First of all, almost all of the datasets are not perfect. Each of the datasets contains a large amount of data, there is certainly a lot of messy and uncleaned data. They are needed to be converted the data files into a readable format. Moreover, using these data straightly without cleaning will do more harm than good and of course will lead to inaccurate conclusions. With that being said, the very next step after importing data files is to clean the data to deal with missing values, incorrect format values, and put away any unnecessary values for building models later. Some of the values are not in its correct forms such as the selling date is presented as text, so the ‘date’ column is converted to correct ‘datetime’ format. There are three columns with missing values: ‘waterfront’, ‘view’, and ‘yr_renovated’. Each of these columns have different types of null values. In the ‘waterfront’ column, there are only two unique values: 0 is for a house with no waterfront view and 1 is for a house with a waterfront view. About 88% of the data in this column has a value of 0 and about 11% of the data is null values. Therefore, the missing values in 'waterfront' column are replaced with the value of 0. On the other hand, most of the values (about 90%) in the ‘view’ have a value of 0 and there is only 0.29% of the column data is null values. Hence, the NaN values in this column are replaced with 0. Lastly, most of the values (about 79%) in the 'yr_renovated' columns are 0 which means that the renovated year for most of the houses are unknown. It is best to drop the column as replacing NaN with the median of the column is useless. In the ‘sqft_basement’ column, more than half of the values (~59%) is 0 which means no basement and only 2% of the data is unknown as indicated as '?'. Since the majority of the data is 0 or unknown ('?'), it is best to change this data into a binary data: 0 for no basement and 1 for having a basement.

After dealing with data type format and missing values, I also need to check and deal with any outliers and duplicate values for a better analysis later. The outliers are checked by using histograms to visualize the distribution of each column. The outliers in ‘bedrooms’ and ‘price’ are cut off where the graphs start to skew. For checking duplicate values, the 'id' column is relied on because this column contains unique values as it is the unique number to identify each house. Therefore, any existing duplicates in 'id' column are eliminated from the dataset and then the whole ‘id’ column is dropped as it is not beneficial for building models later. After cleaning the dataset, it is saved into a new csv file for later analysis. 

## EXPLORATORY ANALYSIS
I first started to explore the ‘bedrooms’ column and figured out that the mean price of the house start to fluctuate when the number of bedrooms gets to 8 or more. Therefore, I created a separated data frame and limit the number of bedrooms to smaller than or equal to 7. These two data frames are used for later regression comparison to see whether there is a significant improve in modeling. 

```
bedrooms;
1     313389.652632;
2     388020.048943;
3     431077.318624;
4     523458.969920;
5     551074.785948;
6     560732.697561;
7     608828.172414;
8     588444.444444;
9     670999.750000;
10    655000.000000;
11    520000.000000;
Name: price, dtype: float64;
```

In order to have a rough idea of which features have good impacts on home price, taking a look at correlation relationship between each feature and the target ‘price’ and also between each feature would help. The features that are highly correlated to 'price' are: 'grade', 'sqft_living', 'sqft_living15', 'sqft_above', 'lat', 'bathrooms', and 'bedrooms'. However, 'sqft_living' is highly correlated to 'sqft_above', 'sqft_living15', and 'bathrooms'. When building a model later, 'sqft_living' is not incorporated with other three features. However, other features can be incorporated together separately since they are not highly correlated to each other.

Now that I had a rough idea about key features, I took some features and pared it with the target ‘price’ to see the impact:
•	The home price increases as the grade of home increases
•	The number of bathrooms’ increase leads to the increase of home price
•	Similarly, the home price increases as the number of bedrooms increases

![](https://drive.google.com/drive/folders/1C1VNHp_2omWuwfg-CWsfbBm0hfIh_DWc?usp=sharing) 

Since 'price' is our main target and our goal is to see which predictors work best in predicting out target. Therefore, we need to check to see whether the target 'price' is close to a normal distribution by using an KDE plot. The distribution of ‘price’ turned out to be not perfectly normally distributed, so it needed to be transformed by log transformation and saved it to a new column.

```
df['log_price'] = np.log(df['price'])
```

In order to select most appropriate predictors for our model, two methods are used:

#### a.	Stepwise Selection by p-values: 
It starts with an empty model (only including the intercept). Each time, the variable that has an associated parameter estimate with the lowest p-value is added to the model. After each addition, the algorithm will look at the p-values of previously added variables and remove them if the p-value exceeds a certain value. The algorithm stops when no variable can be added or removed. 
After defining and running Stepwise Selection, the resulting features are 'sqft_above', 'sqft_living15', 'basement', 'grade', 'lat', 'yr_built', 'view', 'condition', 'floors', 'bathrooms', 'zipcode', 'sqft_lot', 'sqft_living', 'bedrooms', 'waterfront', and 'long'

#### b.	Recursive Feature Elimination by weight of the coefficients:
Using this functions to select top 5 features that have the highest weight of the coefficients: 'bathrooms', 'waterfront', 'condition', 'grade', and 'lat'.

## CALCULATE OLS REGRESSION RESULTS THROUGH STATSMODELS
#### 1) The 1st model
OLS regression was calculated based on chosen features 'grade', 'sqft_living', 'bedrooms' from the Stepwise Selection method and also based on the high correlation relationships with the target ‘price’. The results showed that The R-squared and Adj.R-squared are not high: only 0.411. In addition, the skewness was negative (-0.088). These values indicated that this model was not a great fit. Therefore, I needed to try other features that are highly correlated with 'price', but also need to drop 'sqft_living' since it is highly correlated to other features to avoid multicollinearity.

#### 2) The 2nd model
OLS regression was calculated based on features 'grade', 'lat', ‘bathrooms’, 'bedrooms'. This time, the R-squared (=0.585), Adj.R-squared (=0.585) and skewness (=0.082) values were clearly improved. As stated earlier, the next model was tested with the number of bedrooms limited to smaller than and equal to 7 to see whether the model was improved.

#### 3) The 3rd model
OLS regression was calculated based on the same features as above: 'grade', 'lat', ‘bathrooms’, 'bedrooms', but with the number of bedrooms limited to smaller than and equal to 7. In this model, nothing really changed. All the important values stayed the same.

#### 4) The 4th model
In this last model, OLS regression was calculated without 'bedrooms' feature because this feature seemed not helping with improving the model. On the other hand, the next recommended features from recursive feature elimination were added: 'condition' and 'waterfront'. The result showed that R-squared, Adj.R-squared are the highest (=0.604), compared to other 3 models. The skewness is also the lowest or nearest to 0 (=0.065) which indicates the normally-distributed errors being symmetrically distributed about the mean. The Kurtosis value is also closest to 3 (=3.482) which indicates its closeness to normal distribution. 

## ASSUMPTIONS CHECK AND MODEL VALIDATION
1.	Normality is checked by using Q-Q plot. This type of plot can help compare error residuals against a standard normal distribution. There are no major deviations from the normal distribution line, proving the earlier assumption of normality.

2.	In order to validate the multivariate linear regression model calculated with statsmodels, train-test-split and cross validation were performed. Train-test- split splits the data into two sections: one serves as the training data set, and the other serves as the testing data set. Each set generates a mean squared error, and the difference between the two will summarize how well the predicted values compare to actual values. 
The train-test-split values calculated for this model had a negligible difference, suggesting the model is an appropriate fit. 

However, the train-test-split models calculates a slightly different mean squared error each time the model is run due to the random split of train and test data. For this reason, K-fold cross validation is a better method used to validate the multivariate linear regression. K-fold cross validation averages the individual results from multiple linear models which each use a different section of the test data set. K-fold, with 5, 10 and 20 partitions, was used to assess the mean squared error as well as the coefficient of determination for the model. The mean squared error came back exactly the same, while the coefficients of determination have very little difference.

3.	After performing train-split test, the distribution of model residuals needed to be checked to ensure it follows a normal distribution and homoscedastic. This test was done by plotting two scatter plots between residuals of training/testing sets and the predicted values of logged price. In addition, the histograms of training/testing sets were also plotted. The results showed that the model residuals are normally distributed and homoscedastic. 

As the model is confirmed to be a great fit, we can state according to the calculation that the R-squared value and the coefficient of determination, is around 0.60 for this model. This means that 60% of the variations in home prices are explained by the independent variables: 'grade', 'lat', 'bathrooms', 'condition', 'waterfront'

![](https://drive.google.com/file/d/1zgQr12q1I6OkYw46BuECCzTg2tI3kv4J/view)

![](https://drive.google.com/file/d/1Tbf5xAlQaHSBUuFZXSYWxmxSxwHubQMJ/view)

## THE FINAL MODEL 
 After performing all the tests to ensure that the chosen model is a great fit, the final model regression formula is done by inserting in all the corresponding coefficients of chosen features ('grade', 'lat', 'bathrooms', 'condition', 'waterfront') and the intercept from calculated OLS regression summary table:
#### ln(price) = grade(0.207) + latitude(1.397) + bathrooms(0.114) + condition(0.099) + waterfront(0.602) - 55.58

## COEFFICIENT INTERPRETATION
The model was fitted in form of log(price). Therefore, the general interpretation for the log-transformed variable is as: 1 unit increase in X will lead to (exp(coefficient)-1)*100%.
•	The coefficient of grade is 0.207. This can be interpreted as for one-unit increase in grade, the price of the house will increase by 23% (e0.207 = 1.23) ((1.23-1) 100) while keeping the other predictors constant
•	The coefficient of the latitude is 1.397. This can be interpreted as for 1 unit increases in latitude, the price of the house will increase by 304% (e**(1.397) = 4.04)
•	The coefficient of the bathrooms is 0.114. This can be interpreted as for 1 unit increases in the number of bathrooms, the price of the house will increase by 12% (e**(0.114) = 1.12)
•	The coefficient of the condition is 0.099. This can be interpreted as for 1 unit increases in the condition, the price of the house will increase by 10% (e**(0.099) = 1.10)
•	The coefficient of the waterfront is 0.602. This can be interpreted as for having a waterfront view, the price of the house will increase by 83% (e**(0.602) = 1.83)

## FUTURE ANALYSIS	
There are a lot of other factors that can be used to interpret the house price from the dataset. In addition, the features that were used in this analysis were also limited to some extent. For instance, the price of the house was limit to less than $950,000. Therefore, based on the needs of the customers, the model can be customized with different features such as sale time (which month or season), the age of the house, renovation year, number of floors, and so on.

## CONCLUSION
To sum up from the analysis, the top three features that have great impact on house sale price are the location, the waterfront view, and the grade of the house. Two out of these three features are belong to where you should purchase a house. Renovating or upgrading the house will increase the value of the house, but of course, not as important as the location and whether it has a waterfront view. In addition, when coming to purchase anything, especially something that requires a big budget, people tend to look for some sort of predetermined standard. Therefore, it makes sense that the grade of the house which is determined based on King County grading system has a great impact on the price of a house, just after the location and the waterfront view. The higher the grade, the more willing people are to spend money and buy the house. Hence, the value of the house increases. 

