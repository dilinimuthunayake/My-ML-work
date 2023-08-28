#!/usr/bin/env python
# coding: utf-8

# In[71]:


#import all the required libraries
import pandas as pd  # working with data, Data processing, data analysis, reading CSV files, working with dataframe
import numpy as np # work with mathematical operations 
from matplotlib import pyplot as plt # Data visualization and graphical plotting
import seaborn as sns # to visualize random distributions/statistical graphics

#These commands set options for displaying DataFrames in pandas, 
#a popular Python library for data manipulation and analysis. 
#The first command sets the maximum number of rows to be displayed to 500, 
#the second command sets the maximum number of columns to be displayed to 500, 
#and the third command sets the maximum width of the display to 1000 characters.
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#This code imports the "warnings" module and then uses the filterwarnings()
#function to ignore any warnings that are generated. 
#This means that if any code you run generates a warning, it will not be displayed.
import warnings
warnings.filterwarnings('ignore')


# In[72]:


#data sets 
df_train=pd.read_csv('train.csv')
df_center_info = pd.read_csv('fulfilment_center_info.csv')
df_meal_info = pd.read_csv('meal_info.csv')


# In[73]:


df_train.head()


# In[74]:


df_train.dtypes


# In[75]:


df_center_info.dtypes


# In[76]:


df_meal_info.dtypes


# In[77]:


df_center_info.head()


# In[78]:


df_meal_info.head()


# In[79]:


df_train.info()


# In[80]:


df_meal_info.info()


# In[81]:


df_center_info.info()


# In[82]:


#merge the train table with the center_info table on center_id
df_train=pd.merge(df_train,df_center_info,how='left',left_on='center_id',right_on='center_id') 


# In[83]:


#merge the train table with the meal_info table on meal_id
df_train=pd.merge(df_train,df_meal_info,how='left',left_on='meal_id',right_on='meal_id')


# In[84]:


df_train.head()


# In[85]:


# Convert 'city_code' and 'region_code' into a single feature - 'city_region'.
#make it simpler , reduce overfitting(poor performance of new data,cuz trained too muvh on training ) , increase perfomance
df_train['city_region'] =         df_train['city_code'].astype('str') + '_' +         df_train['region_code'].astype('str')


# In[86]:


df_train.dtypes


# In[87]:


df_train.isna().sum()


# In[88]:


#Label encode -form of feature engineering ,categorical columns in to numerical for use in regression
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
label_encode_cols=['center_id','meal_id','city_code','region_code','city_region','center_type','category','cuisine']


# In[89]:


#for loop intergrated (get together with same caracteritcs ) the list of collums and create new collumns
#fit() learn the unique classes in the collum
#transform () to convert
for col in label_encode_cols:
    le.fit(df_train[col])
    df_train[col+'_encoded']=le.transform(df_train[col])


# In[90]:


# Feature engineering technique - treat 'week' as a cyclic feature.
# Encode it using sine and cosine transform.
#new columns that represent the sine and cosine of week of the year. 
#improve the accuracy,cyclical patterns
# 52.143 is used to normalize ='Week' collumn Values to 0 nd 1 
#mprove perfomance /reduce dominating variaes,normalization 
df_train['week_sin'] =         np.sin(2 * np.pi * df_train['week'] / 52.143)
df_train['week_cos'] =         np.cos(2 * np.pi * df_train['week'] / 52.143)


# In[91]:


# Feature engineering - % difference between base price and checkout price.
#pricing of the meals and how much discounts are offered.
df_train['price_diff_percent'] =         (df_train['base_price'] - df_train['checkout_price']) /         df_train['base_price']


# In[92]:


# Feature engineering - Convert email and homepage features into a single feature - 'email_plus_homepage'.
#the overall promotion received by a meal.
df_train['email_plus_homepage'] =         df_train['emailer_for_promotion'] +         df_train['homepage_featured']


# In[93]:


df_train.head(2)


# In[96]:


# Prepare a list of columns to train on.
# Also decide which features to treat as numeric and which features to treat
# as categorical.
columns_to_train=['week',
                  'week_sin',
                  'week_cos',
                  'cuisine_encoded',
                  'category_encoded',
                  'center_type_encoded',
                  'email_plus_homepage',
                  'price_diff_percent',
                  'city_region_encoded',
                  'meal_id_encoded',
                  'center_id_encoded',
                  'op_area',
                  'base_price',
                  'checkout_price']


# In[97]:


#list of categorically encoded columns
categorical_columns=['email_plus_homepage',
                       'city_region_encoded',
                       'center_type_encoded',
                       'category_encoded',
                       'cuisine_encoded',
                       'center_id_encoded',
                       'meal_id_encoded']


# In[98]:


#list of numerical columns
numerical_columns=[col for col in columns_to_train if col not in categorical_columns ]


# In[99]:


#, each list is used to store a specific metric for the 
#machine learning models that will be trained.
MODELS = []
MAE = []
MSE = []
RMSE = []
RMSLE = []
DIFFERENCE = []
R2 = []


# In[100]:


# Log transform the target variable - num_orders.
#useful for handling positive data that has a skewed distribution
#improve the model accuracy, make the data more normal and reduce outliers

df_train['num_orders_log1p'] = np.log1p(df_train['num_orders'])


# In[102]:


# Train-Test split.
#X - Input features, y-targret varibale 
from sklearn.model_selection import train_test_split
X = df_train[categorical_columns + numerical_columns]
y = df_train['num_orders_log1p']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    shuffle=False)


# In[103]:


# Standard Scaling - Removing mean from each value and dividing by the standard deviation.(reduce outliers)
#important for some machine learning algorithms that are sensitive to the scale of the input
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[104]:


#standardizes the data by removing the mean and scaling to unit variance

X_train = sc.fit_transform(X_train)


# In[105]:


#sc object contains the mean of each feature computed from the training data.
#mean of the feature
sc.mean_


# In[106]:


#object contains the variance of each feature computed from the training data.
#variance is the square of the standard deviation
#variance of the standardized data will be 1
#variance of the feature
sc.var_


# In[107]:


#standardizes the data by removing the mean and scaling to unit variance
X_test = sc.transform(X_test)


# In[143]:


df_train.info()


# In[144]:


df_train.isna().sum()


# In[108]:


pip install catboost


# In[109]:


pip install lightgbm


# In[110]:


pip install xgboost


# In[111]:


from lightgbm import LGBMRegressor, LGBMClassifier, Booster


# In[112]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# In[113]:


dtr=DecisionTreeRegressor()
rfr=RandomForestRegressor()
adb=AdaBoostRegressor()
cbr=CatBoostRegressor()
lgbmr=LGBMRegressor()
xgbr=XGBRegressor()


# In[114]:



dtr.fit(X_train,y_train)
MODELS.append('DecisionTree Regression')
y_pred_dtr=dtr.predict(X_test)
rfr.fit(X_train,y_train)
MODELS.append('Random Forest Regression')
y_pred_rfr=rfr.predict(X_test)
adb.fit(X_train,y_train)
MODELS.append('AdaBoost Regression')
y_pred_adb=adb.predict(X_test)
cbr.fit(X_train,y_train)
MODELS.append('CatBoost Regression')
y_pred_cbr=cbr.predict(X_test)
lgbmr.fit(X_train,y_train)
MODELS.append('LGBM Regression')
y_pred_lgbmr=lgbmr.predict(X_test)
xgbr.fit(X_train,y_train)
MODELS.append('XGB Regression')
y_pred_xgbr=xgbr.predict(X_test)


# In[115]:


from sklearn.ensemble import VotingRegressor


# In[116]:


#Average Ensemble
# Fitting Voting Regressor to the Training set
avgrc = VotingRegressor(estimators=[('Random Forest',rfr),
                                 ('CatBoost',cbr)])
avgrc.fit(X_train, y_train)
MODELS.append('Average Ensemble 001')


# In[117]:


y_pred_avgrc= avgrc.predict(X_test)
y_pred_avgrc


# In[118]:


#Average Ensemble 2
# Fitting Voting Regressor to the Training set
avgrcl = VotingRegressor(estimators=[('Random Forest',rfr),
                                 ('CatBoost',cbr),
                                 ('LGBM Regression',lgbmr)])
avgrcl.fit(X_train, y_train)
MODELS.append('Average Ensemble 002')


# In[119]:


y_pred_avgrcl= avgrcl.predict(X_test)
y_pred_avgrcl


# In[120]:


#Average Ensemble 3
# Fitting Voting Regressor to the Training set
avgrclx = VotingRegressor(estimators=[('Random Forest',rfr),
                                 ('CatBoost',cbr),
                                 ('LGBM Regression',lgbmr),
                                 ('XGB Regression',xgbr)])
avgrclx.fit(X_train, y_train)
MODELS.append('Average Ensemble 003')


# In[121]:


y_pred_avgrclx= avgrclx.predict(X_test)
y_pred_avgrclx


# In[122]:


MODELS


# In[123]:


#Evaluation through metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_squared_log_error,r2_score


# In[124]:


#Decision Tree
mse = mean_squared_error(y_test,y_pred_dtr)
mae= mean_absolute_error(y_test,y_pred_dtr)
rmsle= mean_squared_log_error(y_test,y_pred_dtr)
print('MSE: ',mse)
rmse = mse**0.5
print('RMSE: ',rmse)
print('MAE: ',mae)
print('RMSLE: ',rmsle)
diff=abs(mae-rmse)
print('Difference :',diff)
MAE.append(mae)
MSE.append(mse)
RMSE.append(rmse)
RMSLE.append(rmsle)
DIFFERENCE.append(diff)
r2 = r2_score(y_test, y_pred_dtr)
print('R2: ',r2)
R2.append(r2)


# In[125]:


#RandomForest
mse = mean_squared_error(y_test,y_pred_rfr)
mae= mean_absolute_error(y_test,y_pred_rfr)
rmsle= mean_squared_log_error(y_test,y_pred_rfr)
print('MSE: ',mse)
rmse = mse**0.5
print('RMSE: ',rmse)
print('MAE: ',mae)
print('RMSLE: ',rmsle)
diff=abs(mae-rmse)
print('Difference: ',diff)
MAE.append(mae)
MSE.append(mse)
RMSE.append(rmse)
RMSLE.append(rmsle)
DIFFERENCE.append(diff)
r2 = r2_score(y_test, y_pred_rfr)
print('R2 :',r2)
R2.append(r2)


# In[126]:


#AdaBoost
mse = mean_squared_error(y_test,y_pred_adb)
mae= mean_absolute_error(y_test,y_pred_adb)
rmsle= mean_squared_log_error(y_test,y_pred_adb)
print('MSE: ',mse)
rmse = mse**0.5
print('RMSE: ',rmse)
print('MAE: ',mae)
print('RMSLE: ',rmsle)
diff=abs(mae-rmse)
print('Difference :',diff)
MAE.append(mae)
MSE.append(mse)
RMSE.append(rmse)
RMSLE.append(rmsle)
DIFFERENCE.append(diff)
r2 = r2_score(y_test, y_pred_adb)
print('R2: ',r2)
R2.append(r2)


# In[127]:


#CatBoost
mse = mean_squared_error(y_test,y_pred_cbr)
mae= mean_absolute_error(y_test,y_pred_cbr)
rmsle= mean_squared_log_error(y_test,y_pred_cbr)
print('MSE: ',mse)
rmse = mse**0.5
print('RMSE: ',rmse)
print('MAE: ',mae)
print('RMSLE: ',rmsle)
diff=abs(mae-rmse)
print('Difference :',diff)
MAE.append(mae)
MSE.append(mse)
RMSE.append(rmse)
RMSLE.append(rmsle)
DIFFERENCE.append(diff)
r2 = r2_score(y_test, y_pred_cbr)
print('R2 :',r2)
R2.append(r2)


# In[128]:


#LightGBM
mse = mean_squared_error(y_test,y_pred_lgbmr)
mae= mean_absolute_error(y_test,y_pred_lgbmr)
rmsle= mean_squared_log_error(y_test,y_pred_lgbmr)
print('MSE: ',mse)
rmse = mse**0.5
print('RMSE: ',rmse)
print('MAE: ',mae)
print('RMSLE: ',rmsle)
diff=abs(mae-rmse)
print('Difference: ',diff)
MAE.append(mae)
MSE.append(mse)
RMSE.append(rmse)
RMSLE.append(rmsle)
DIFFERENCE.append(diff)
r2 = r2_score(y_test, y_pred_lgbmr)
print('R2: ',r2)
R2.append(r2)


# In[129]:


#xgBoost
mse = mean_squared_error(y_test,y_pred_xgbr)
mae= mean_absolute_error(y_test,y_pred_xgbr)
rmsle= mean_squared_log_error(y_test,y_pred_xgbr)
print('MSE: ',mse)
rmse = mse**0.5
print('RMSE: ',rmse)
print('MAE: ',mae)
print('RMSLE: ',rmsle)
diff=abs(mae-rmse)
print('Difference: ',diff)
MAE.append(mae)
MSE.append(mse)
RMSE.append(rmse)
RMSLE.append(rmsle)
DIFFERENCE.append(diff)
r2 = r2_score(y_test, y_pred_xgbr)
print('R2: ',r2)
R2.append(r2)


# In[130]:


#Average Ensemble 
mse = mean_squared_error(y_test,y_pred_avgrc)
mae= mean_absolute_error(y_test,y_pred_avgrc)
rmsle= mean_squared_log_error(y_test,y_pred_avgrc)
print('MSE: ',mse)
rmse = mse**0.5
print('RMSE: ',rmse)
print('MAE: ',mae)
print('RMSLE: ',rmsle)
diff=abs(mae-rmse)
print('Difference: ',diff)
MAE.append(mae)
MSE.append(mse)
RMSE.append(rmse)
RMSLE.append(rmsle)
DIFFERENCE.append(diff)
r2 = r2_score(y_test, y_pred_avgrc)
print('R2: ',r2)
R2.append(r2)


# In[131]:


#Average Ensemble 2
mse = mean_squared_error(y_test,y_pred_avgrcl)
mae= mean_absolute_error(y_test,y_pred_avgrcl)
rmsle= mean_squared_log_error(y_test,y_pred_avgrcl)
print('MSE: ',mse)
rmse = mse**0.5
print('RMSE: ',rmse)
print('MAE: ',mae)
print('RMSLE: ',rmsle)
diff=abs(mae-rmse)
print('Difference: ',diff)
MAE.append(mae)
MSE.append(mse)
RMSE.append(rmse)
RMSLE.append(rmsle)
DIFFERENCE.append(diff)
r2 = r2_score(y_test, y_pred_avgrcl)
print('R2: ',r2)
R2.append(r2)


# In[132]:


#Average Ensemble 3
mse = mean_squared_error(y_test,y_pred_avgrclx)
mae= mean_absolute_error(y_test,y_pred_avgrclx)
rmsle= mean_squared_log_error(y_test,y_pred_avgrclx)
print('MSE: ',mse)
rmse = mse**0.5
print('RMSE: ',rmse)
print('MAE: ',mae)
print('RMSLE: ',rmsle)
diff=abs(mae-rmse)
print('Difference: ',diff)
MAE.append(mae)
MSE.append(mse)
RMSE.append(rmse)
RMSLE.append(rmsle)
DIFFERENCE.append(diff)
r2 = r2_score(y_test, y_pred_avgrclx)
print('R2: ',r2)
R2.append(r2)


# In[136]:


#Convert the metrics values into a data table
model_dict = {'Models': MODELS,
             'Mean Squared Error': MSE,
             'Mean Absolute Error': MAE,
             'Root Mean Squared Error': RMSE,
             'Root Mean Squared Log Error': RMSLE,
             'Difference': DIFFERENCE,
              'R^2': R2}


# In[137]:


model_df = pd.DataFrame(model_dict)
model_df


# In[138]:


model_df = model_df.sort_values(['Mean Squared Error','Mean Absolute Error','Root Mean Squared Error','Root Mean Squared Log Error','Difference','R^2'],
                               ascending=(True,True,True,True,True,False))


# In[139]:


model_df


# In[140]:


best_model = model_df['Models'].values[0]
best_model


# In[141]:


import pandas as pd
import matplotlib.pyplot as plt

# Assign data to variable
data = {'Models': ['DecisionTree Regression', 'Random Forest Regression', 'AdaBoost Regression', 'CatBoost Regression', 'LGBM Regression', 'XGB Regression', 'Average Ensemble 001', 'Average Ensemble 002', 'Average Ensemble 003'],
        'Mean Squared Error': [0.675846, 0.35385, 0.790107, 0.302691, 0.341209, 0.478162, 0.300024, 0.296707, 0.314411],
        'Mean Absolute Error': [0.616843, 0.457961, 0.732386, 0.434451, 0.462614, 0.550644, 0.428497, 0.429375, 0.445132],
        'Root Mean Squared Error': [0.822099, 0.594853, 0.888879, 0.550173, 0.584131, 0.691493, 0.547744, 0.544708, 0.560724],
        'Root Mean Squared Log Error': [0.024829, 0.012415, 0.025623, 0.010894, 0.012021, 0.016545, 0.010703, 0.010592, 0.011214],
        'Difference': [0.205255, 0.136892, 0.156494, 0.115722, 0.121517, 0.140849, 0.119247, 0.115334, 0.115592],
        'R^2': [0.513379, 0.745222, 0.43111, 0.782057, 0.754324, 0.655715, 0.783978, 0.786366, 0.773619]}

# Convert to dataframe
data = pd.DataFrame(data)


import seaborn as sns

sns.heatmap(data.iloc[:,1:],annot=True,annot_kws={"size": 8})
plt.title('Heat map -  Evaluation for all the models')
plt.show()


# In[148]:


# Get predictions on the test set and prepare submission file.
X = df_train[categorical_columns + numerical_columns]

pred = avgrc.predict(X)
pred = np.expm1(pred)

submission_df = df_train.copy()
submission_df['num_orders'] = pred
submission_df = submission_df[['id', 'num_orders']]
submission_df.to_csv('submission.csv', index=False)

import os
print(os.getcwd())


# In[149]:


import pickle

# Train your model 
'model = Average Ensemble 002'

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[ ]:




