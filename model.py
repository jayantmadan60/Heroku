# Setting working directory
import os
#print(os.listdir())

# importing necessary modules
import pandas as pd
import numpy as np

# read CSV file
df = pd.read_csv('50_Startups.csv')
#print(df.columns)
print (df.head())

# Exploratory data analysis
#print(df.info())
#print('\n=================================================================\n')
#print(df.describe().T)

# checking null values if any
#print(df.isna().sum())


# checking for multicollinearity through VIF
correlation = pd.DataFrame(1/(1-df.corr()))
#print(correlation)

# removing outlier function
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

# removed outlier from profit column and dropping categorical column
df = df[(df[['R&D Spend', 'Administration', 'Marketing Spend','Profit']] != 0).all(axis=1)]
df = remove_outlier(df,'Profit')
df = df.drop(['State','Administration'],axis = 1)


#Normalization/scaling of data - understanding scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)



#converting data back to pandas dataframe
MyData_scaled = pd.DataFrame(scaled_data)
MyData_scaled.columns = ['R&D Spend','Marketing Spend','Profit']



#Separating features and response
features = ['R&D Spend','Marketing Spend']
response = ["Profit"]
X=MyData_scaled[features]
y=MyData_scaled[response]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Importing neccesary packages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Fitting lineaar regression model
model = LinearRegression()
model.fit(X_train, y_train)


#Checking accuracy on test data
accuracy = model.score(X_test,y_test)
print(accuracy*100,'%')

y_pred = model.predict(X_test)
print (r2_score(y_test,y_pred)) #predcited values on test data


#Dumping the model object
import pickle
pickle.dump(model, open('model.pkl','wb'))


#Reloading the model object
model = pickle.load(open('model.pkl','rb'))
print('------------')
print(model.predict([[162597.70, 443898.53]]))




























































