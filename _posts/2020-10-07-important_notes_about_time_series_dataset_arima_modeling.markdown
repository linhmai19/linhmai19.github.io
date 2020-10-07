---
layout: post
title:      "IMPORTANT NOTES ABOUT TIME SERIES DATASET – ARIMA MODELING"
date:       2020-10-07 16:23:14 +0000
permalink:  important_notes_about_time_series_dataset_arima_modeling
---


Time series is an ordered sequence of numerical data points of, usually, an independent variable at equally spaced time intervals. There is no minimum or maximum amount of time that must be included, allowing the data to be gathered in a way that provides the information being sought by the investor or analyst examining the activity and make a forecast for the future. Some of the examples of time series data include changes in stock prices, oil flow through a pipeline, utility studies, and so on.
In this article, I will talk about some important points that I think you should pay attention to when working with time-series data.

### 1. The ‘date’ Column

The very first time when starting to work with time-series data is the format of the date column. The ‘date’ column needs to be in the ‘datetime’ format, so it can be understood by Python when doing modeling later. In addition, the ‘date’ column needs to be also set as the index of the dataset to help with plotting for visualization.
```
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')
df.set_index('date', inplace=True)
```

### 2. Layout of the Feature Columns

When working with time series, you may encounter some of the datasets that have the actual time series values store in separate columns, next to other information/feature columns. This type of storing is called Wide Format. With this layout, it is very easy for the reader to review and interpret the data. However, when coming to machine learning, we need to know exactly the name of the column that the data can be found. Because columns’ names are metadata, the algorithms will miss out on what dates each value is for. Therefore, to use the ARIMA model later, we need to reshape the dataset from Wide Format to Long Format. To do this, we can define a function using ‘melt’ of pandas:
```
def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionName', 'City', 'State', 
                     'Metro', 'CountyName'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], 
                                    infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})
```

### 3. Trend, Seasonality, and Noise

Most of the time series data can be decomposed into three major properties: trend, seasonality, and noise. Simply decomposing the series can itself yield some valuable insights:

* Trend refers to the underlying trend of metrics. For example, a website that has an increase in popularity shows an upward trend line.
* Seasonality refers to patterns that repeat within a fixed period. For instance, a website has more visits on weekends than on weekdays, or electricity consumption is high during the day and low during the night.
* Noise (or also called Random, Irregular, Remainder) is the residual of the original time series after the seasonality and trend are removed.

These three components can be detected by using ‘decomposition’ from Statsmodel:
```
decomposition = seasonal_decompose(np.log(df_melt))

# Gather the trend, seasonality and noise of decomposed object
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot gathered statistics
plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(np.log(df_melt), label='Original', color="blue")
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend', color="blue")
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality', color="blue")
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals', color="blue")
plt.legend(loc='best')
plt.tight_layout();

# Drop missing values from residuals
dfmelt_log_decomp = residual
dfmelt_log_decomp.dropna(inplace=True)
```

### 4. Stationarity

Stationarity means that the statistical properties of a process generating a time series do not change over time. It does not mean that the series does not change over time, just that the way it changes does not itself change over time. Most time series models require the data to be stationary. A time series is said to be stationary if its statistical properties such as mean, variance, and covariance remain constant over time. We check for stationarity by using visual analysis and performing a statistical test. We can check for stationarity of a time series by defining a function with Rolling Statistics and Dickey-Fuller Test:
```
def stationarity_check(TS):    
    # Calculate rolling statistics
    roll_mean = TS.rolling(window=8, center = False).mean()
    roll_std = TS.rolling(window=8, center = False).std()
    
    # Perform the Dickey-Fuller Test
    dftest = adfuller(TS['value']) 
    
    # Plot rolling statistics
    fig = plt.figure(figsize=(12,6))
    plt.plot(TS, color='blue',label='Original')
    plt.plot(roll_mean, color='red', label='Rolling Mean')
    plt.plot(roll_std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    
    # Print Dickey-Fuller test results
    print ('Results of Dickey-Fuller Test: \n')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic',
                                'p-value',
                                '#Lags Used',
                                'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    return None
```

The output will show the p-value. If this p-value is higher than 0.05 significance value, the null hypothesis fails to be rejected which means the time series dataset is not stationary. There are ways to make a time series stationary such as tunning hyperparameters (p-lag order, d-egree of differencing, q-order of moving average).

### 5. Autoregressive Integrated Moving Average (ARIMA) modeling

The fitting of time series models can be an ambitious undertaking. There are many methods of model fitting including ARIMA models, Box-Jenkins Multivariate models, Holt-Winters Exponential Smoothing (single, double, triple). In this article, we will only focus on talking about the ARIMA models. ARIMA modeling is a way of forecasting the future values of the time series by using previous values. There are three components for an ARIMA model. An ARIMA model has order (p, d, q) where p, d, q refer to how many terms we consider for each of the model’s three components:

**a. Auto Regression:**
Calculate the next term based on the values of the last p terms, using an individually defined coefficient for each of those terms. This is basically a regression of the time series onto itself

**b. Integrated Term:**
If this term is non-zero, we transform the time series from its original values to the difference between values at time t, and time t-d. The parameter d represents the number of differences required to make the series stationary. This is used in cases where we have a general drift in the data.

**c. Moving Average:**
Calculate the next term based on the error from the moving average of the last q terms. This takes a parameter q which represents the biggest lag after which other lags are not significant on the autocorrelation plot.

* Function to evaluate an ARIMA model for a given order (p,d,q)
```
def evaluate_arima_model(X, arima_order):
    # Prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # Make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # Calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error
```

* Function to evaluate combinations of p, d, and q for an ARIMA model
```
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg
```

* An example of function to loop through all zip codes in Harris County, TX to give the best model based on the smallest MSE value
```
roi_list = {}
for index, x in df[(df['CountyName'] == 'Harris') & 
                   (df['State'] == 'TX')].iterrows():
    print(x['RegionName'])
    series = melt_data(df.loc[[index]])
    # evaluate parameters
    p_values = [0, 1, 2]
    d_values = range(0, 2)
    q_values = range(0, 2)
    warnings.filterwarnings("ignore")
    
    order = evaluate_models(series.values, p_values, 
                            d_values, q_values)
    
    model= ARIMA(series, order=order)
    model_fit= model.fit()
    thirty_six_months = model_fit.forecast(steps=36)[0][-1]
    today = series.iloc[-1].value
    roi = (thirty_six_months - today)/today
    roi_list[x['RegionName']] = roi
```

From there, we can sort the list from highest to lowest ROI (based on the MSE values) to choose the top zip codes for investment.

