# Prediction On Hang Seng Index Using Deep Learning
<sub>> This project is my dissertation of the master degree
> Whole project is implemented by Python.</sup>

***Project Objective***
The aim of this research is to use deep learning as the main method to perform stock market forecasting with various model settings such as features engineering, ensemble methods, to provide predictive insights for client trading department, e.g. investment advisors and relationship managers.

The objectives of this research are listed as below:
•	To conduct literature review on stock price prediction / forecasting techniques and cutting edge methodologies and frameworks.
•	To investigate the state of art in data application design and development with a deep learning engine (e.g. Recurrent Neural Network, Long Short Term Memory, and Gated Recurrent Unit etc.). 
•	To investigate the stock market information for features, not only Open, High, Low, Close, Volume and Date, e.g. technical indicators or macroeconomic indicators.
•	To develop a deep learning framework to perform prediction (specify time window, cross-validation, time period and the outcome).
•	To develop a deep learning model to achieve the prediction outcome.
•	To obtain client’s evaluation on delivered product.

***Basic Workflow***
![Work Flow](https://user-images.githubusercontent.com/52281668/181567112-65d9b93f-b5c0-4ed8-8518-d11daeae1c9b.jpg)

***Feature Engineering***
This section is divided into two parts: 1.) various types of technical indicators and 2.) specific types of technical indicators. All of the technical indicators are adopted package named TA-Lib, which using prices open, high, low, close, and volume to create technical indicators. 

•	Type I: Various Types of Technical Indicators
  This type of feature construction includes 58 various technical indicators with total 361 extra features, which some of them are included different time periods. These are concluded to be six aspects of technical indicators:
  -	Volume Indicators
  -	Price Transforms
  -	Momentum Indicators
  -	Cycle Indicators
  -	Overlap Studies
  -	Volatility Indicators
  Type I feature construction is purposed to create plenty of features for feature selection, which based on two machine learning methods (Extreme Gradient Boosting and Support Vector Regression). 

•	Type II: Specific Types of Technical Indicators

  This Type II feature construction adopts 10 various technical indicators, but each type of technical indicators only applies to one model, i.e. one of the data inputs. Therefore, it will be ten more various models for the results comparison. Those ten technical indicators includes:
  -	Average True Range (ATR)
  -	Commodity Channel Index (CCI)
  -	Exponential Moving Average (EMA)
  -	Money Flow Index (MFI)
  -	Momentum (MOM)
  -	Rate of Change (ROC)
  -	Relative Strength Index (RSI)
  -	Simple Moving Average (SMA)
  -	Williams' %R (WILLR)
  -	Weighted Moving Average (WMA)

  Most of the technical indicators also adopted in Type I (except Rate of Change) and the time period has amended from 10 different ranges to be 14.

***Data Cleansing***
The technical indicators (feature engineering) used a range of days of data to generate. Due to a fixed review period (between 01Jan2013 and 31May2020), there are some null (NA) values. The earliest date is different in various models.

"""Normalization (Scaling Data)***
All of the columns in the datasets (Date is index) are normalized as within the range between zero and one (no negative values). This normalization is adopted MinMaxScaler function in scikit-learn (package name).
For the models of type II, the datasets are ready to transform as the inputs of deep learning models. Besides, one more procedures have to be executed for the models of type I – Feature Selection.

***Feature Selection***
This section is only applied on the datasets of Type I methods. Ten out of those additional features (technical indicators) are chosen as the highest importance by using Extreme Gradient Boosting and Support Vector Regression separately. Another words, the prices of open, high, low, close and volume are filtered out in this section.
Before performing feature importance, the dataset will be split into technical indicators, y (next date close price), training and testing with total 4 datasets on each method, which the ratio is 90:10. No shuffling split is applied such that the testing dataset is the latest trading days.

***Training and Testing Datasets***
The ratio of training and testing is 90:10. The deep learning model contains two inputs, one for basic prices (open, high, low, close), another for technical indicators (including Type I and Type II).
Other than that, the correlation plots of these datasets (total 12) indicates that volume is almost negatively correlated with other features. Therefore, all of the models are filtered out volume feature. 

  *Data Generator*
  Using sequential data (numbers of days) is better for Long Short Term Memory. This research only used fixed time frame to assure the model results not affected by different time frame. The time frame is set to be 50 days in one data frame and all are applied to two inputs and y (prediction column).

  *Splitting Datasets*
  Including two inputs (ohlc and technical indicators), prediction column (y) and training & testing, there are totaling six datasets with ratio 90 (training) :10 (testing). Since some features of technical indicators require longer time periods of data (due to formulas), the numbers of rows in various technical indicators / methods are different (Table 3.3.7.2). The dataset is sorted by date from oldest to newest, such that the testing dataset included the index price in 2020, which has a sudden drop and big v-cut rebound (Figure 3.3.7.3). It is a challenge for the deep learning model. A comparison model (OHLC) is a single input with prices of open, high, low and close.
  ![image](https://user-images.githubusercontent.com/52281668/181573387-9649b8d1-a913-468d-aa59-ca1899bcadcc.png)


***Models in Tensorflow.Keras***
After the launch of Tensorflow version 2.0, the Keras API can be implemented in Tensorflow. For the deep learning model in Keras, there are several parts which have to be set up. 
![image](https://user-images.githubusercontent.com/52281668/181571853-a3f4b5a7-948d-4a14-a924-1a8366aa7d28.png)
![image](https://user-images.githubusercontent.com/52281668/181573255-6e0f468e-ede8-4719-a2af-fa68cfb02884.png)

•	Model
Keras model provides two basic APIs: Sequential API and Functional API. This research is adopted Functional API due to more than one inputs, including dataset of Open, High, Low, & Close and dataset of technical indicators. This model type contains inputs ([ohlcv_in and technical_in]) and output (out). 
It is noted that shape is related to the actual dimensions of the datasets. The actual shape is a 3-D array because one row includes the columns of features, and data of 50 consecutive trading days.
Output has been concatenated two outputs from two inputs ([ohlcv_in and technical_in]). Then it used two Dense layers to generate predicted value.
•	Layers
This part only includes three components: Long Short Term Memory (LSTM), Dense, and Dropout. LSTM and Dense layers are categorized as neural networks. Dropout helps for prevention of overfitting, which randomly turn inputs as zero values at a certain level.
•	Activation Functions
Activation function is the calculation method on the neurons. The research is adopted relu, sigmoid and linear functions.
•	Compile
In this research, the compile function consists of three parameters: loss function, optimizer and metrics. Loss is set to be mean-square-error, which is a standard  of self-adjusting weight for the neural network model. Optimizer is set to be Adam with learning rate 0.0005, where it is an algorithm to control the gradient descent issues. Metrics is set to be mean-absolute-error, which is one of the evaluation methods. 
It is noted that these loss function and metrics are related to regression, which predicting numeric values. The classification is totally different settings for these two parameters.
•	Callbacks
In Keras API, there are customized callbacks and standard callbacks (e.g. EarlyStooping, and modelCheckpoint etc.) This research adopted tensorboard callbacks to visualize the results during training process. It is also one of the advantages to use Google Colab as the platform.
•	Fitting The Model
The deep learning model is used training datasets (X_ohlcv_train and X_technical_train) for fitting process. 
-	The datasets has been split out ten percent for validation purpose. 
-	The model has been executed 100 times (epochs=100) with batch size 32. 
-	The data in batch chunks is shuffled. 
-	Tensorboard callbacks is simultaneously activated to record the training and validation process.

***Evaluation Methods***
This research used three parts to perform evaluation: 1) Model Fitting Process, 2) Model Evaluation and 3) Comparison Between Actual and Predicted Values. 
•	Model Fitting Process
To handle the overfitting and underfitting issues, machine learning is mainly to propose comparison between training datasets and testing datasets. Deep learning (neural networks) is slightly different to machine learning, i.e. keep tracks to loss function and metrics on each epoch via model fitting process. The model automatically takes records on each epoch, using model.history.history. 
![image](https://user-images.githubusercontent.com/52281668/181572757-ae4d579e-3382-41b4-ad1d-30f45d669f6f.png)

•	Tensorboard
Tensorboard is a tool to visualize the model fitting process and this research only adopts the records on epochs, which indicates loss function (mean-square-error) and metrics (mean-absolute-error). The orange line is training and the blue one is validation. Against the weight adjustment (back propagation) on each epoch, both two measures are gradually decreasing, i.e. the model keeps improving by self.
![image](https://user-images.githubusercontent.com/52281668/181572721-c3f4bbe3-cf46-4829-a770-2ee646bb8efb.png)

•	Model Evaluation
Keras model includes an inner function evaluate() to examine testing dataset or other external datasets with same data format. Also, this evaluation generates the mean-square-error and mean-absolute error. The results are compared with the results in model fitting. The model structure will be modified if the results are obviously different.

•Comparison Between Actual and Predicted Values
Keras model also predict values / categories using function predict() / predict_classess(). This research uses predict() for index price prediction on testing dataset. The values predicted has been inversely transformed to be exact scale of index prices due to MinMaxScaler’s transformation in data pre-processing stage.
The actual values and predicted values are merged into one dataframe (OHLC as the example, totaling 175 trading days for testing). 
![image](https://user-images.githubusercontent.com/52281668/181573120-3e57ec3c-438d-4f58-b4db-2e0f7c19bd33.png)

•Statistical Description
This research calculates the differences between actual values and predicted values. Despite the mean-square-error and mean-absolute error, these statistical descriptions helps to identify the accuracy of the prediction models. Also, the absolute differences between minimum and maximum are calculated as “RANGE”.
![image](https://user-images.githubusercontent.com/52281668/181573198-b2f86876-a6d5-4ccb-8338-e58d94d05591.png)

***Classification***
The purpose of classification model is to compare the results with regression model. This classification model adopts support vector regression methods as feature selection. 
For the predict values, this is categorized as two classes (named ‘categ’ column), one for price drop in next day, and zero for price rise in next day. It is the model related to binary classification. It is noted that there is no feature related to next day price, except the class.

•Counting
For the ‘categ’ column, it is counted the days indicating rises (944 days) rather than the days indicating drops (873 days).
![image](https://user-images.githubusercontent.com/52281668/181573797-f9153f0f-d95a-4852-acd1-edb7fe8b9dd2.png)

This classification model adopts support vector regression for feature selection. This dataset contains only one different column correspondent to same methods on regression – prediction column. The importance on features are totally different, which including:
  •	ADXR 14 & 50: Average Directional Movement Index Rating with time period 14 & 50 days
  •	MFI 25 & 50: Money Flow Index with time period 25 & 50 days
  •	PLUS_DI 7: Plus Directional Indicator  with time period 7 days
  •	PLUS_DM 100: Plus Directional Movement with time period 100 days
  •	HT_DCPERIOD: Hilbert Transform - Dominant Cycle Period
  •	KAMA 25: Kaufman Adaptive Moving Average with time period 25 days
  •	NATR 20 & 25: Normalized Average True Range with time period 20 days & 25 days

•Keras Model
It is noted that the prediction column ‘categ’ contains numbers of one and zero (label encoding, 1-D array), which is binary classification. Therefore, one-hot encoding method (2-D array) is not applied to prediction column. 
The structure of deep learning model is basically unchanged. Only the activation functions of final outputs are changed to be two sigmoid functions because sigmoid function is more appropriate for the binary classification. 
Besides, the model compile function is not applied mean-square-error and mean-absolute-error. The loss function is set to be ‘binary_crossentropy’ for binary classification and the metrics is set to be accuracy.

•Evaluation
The inner function model.evaluation() is adopted for testing dataset.
![image](https://user-images.githubusercontent.com/52281668/181574212-d6511e69-3d7a-4afd-ad9d-510fa6a389de.png)
Other than that, confusion matrix and classification report are also adopted in this classification evaluation.
![image](https://user-images.githubusercontent.com/52281668/181574182-075a2e83-98cc-4553-b704-53970c73c2e1.png)
The results of this classification will be shown in Chapter 4 and discussed how it does a counter-example.
![image](https://user-images.githubusercontent.com/52281668/181574159-9d3e3607-371a-444b-9965-4bd9cf2ddfa8.png)
