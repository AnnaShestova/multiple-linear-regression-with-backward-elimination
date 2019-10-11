# multiple-linear-regression-with-backward-elimination
Multiple linear regression model implementation with automated backward elimination (with p-value and adjusted r-squared) in Python and R for showing the relationship among profit and types of expenditures and the states.

###########################################################################################

# Multiple linear Regression with Automated Backward Elimination (with p-value and adjusted r-squared) in Python
###########################################################################################

## Importing libraries    
import numpy as np    
import matplotlib.pyplot as plt   
import pandas as pd   

## Importing dataset      
dataset = pd.read_csv('50_Startups.csv')    
X = dataset.iloc[:, :-1].values   
Y = dataset.iloc[:, 4].values   

## Encoding categorical data    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder   
labelencoder_X = LabelEncoder()   
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])   
onehotencoder = OneHotEncoder(categorical_features = [3])   
X = onehotencoder.fit_transform(X).toarray()    

##Avoiding dummy variable trap    
X = X[:, 1:]    

## Splitting the dataset into the training set and test set   
from sklearn.model_selection import train_test_split    
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)   
  
## Applying linear regression model   
from sklearn.linear_model import LinearRegression   
regressor = LinearRegression()    
regressor.fit(X_train, Y_train)   

## Predicting test set results    
y_pred = regressor.predict(X_test)    

## Creating automated backward elimination function   
import statsmodels.api as sm    
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)   
def backwardElimination(x, SL):   
       numVars = len(x[0])   
       temp = np.zeros((50,6)).astype(int)   
       for i in range(0, numVars):   
           regressor_OLS = sm.OLS(Y, x).fit()    
           maxVar = max(regressor_OLS.pvalues).astype(float)   
            adjR_before = regressor_OLS.rsquared_adj.astype(float)    
            if maxVar > SL:   
                    for j in range(0, numVars - i):   
                   if (regressor_OLS.pvalues[j].astype(float) == maxVar):    
                        temp[:,j] = x[:, j]   
                       x = np.delete(x, j, 1)  
                       tmp_regressor = sm.OLS(Y, x).fit()    
                       adjR_after = tmp_regressor.rsquared_adj.astype(float)   
                       if (adjR_before >= adjR_after):   
                            x_rollback = np.hstack((x, temp[:,[0,j]]))    
                            x_rollback = np.delete(x_rollback, j, 1)    
                            print (regressor_OLS.summary())   
                           return x_rollback   
                        else:   
                           continue    
        regressor_OLS.summary()   
        return x    

## Implementing automated backward elimination    
SL = 0.05   
X_opt = X[:, [0, 1, 2, 3, 4, 5]]    
X_Modeled = backwardElimination(X_opt, SL)    
regressor_OLS = sm.OLS(endog = Y, exog = X_Modeled).fit()   
regressor_OLS.summary()     

