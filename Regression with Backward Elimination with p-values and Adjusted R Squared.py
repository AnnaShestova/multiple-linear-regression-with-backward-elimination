# Multiple Linear Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Avoiding dummy variable trap
X = X[:, 1:]

# Applying multiple linear regression on all features
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print('Model score: '+str(regressor.score(X_test,Y_test)))

# Predicting re results
y_pred = regressor.predict(X_test)

# Creating automated backward elimination function with p-values and adjusted r-sqaured
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
 
# Applying the backward elimination 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
regressor_OLS = sm.OLS(endog = Y, exog = X_Modeled).fit()
regressor_OLS.summary()

# Splitting new dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_Modeled, Y, test_size = 0.2, random_state = 0)

# Applying linear regression model
from sklearn.linear_model import LinearRegression
regressor_new = LinearRegression()
regressor_new.fit(X_train, Y_train)
print('Model score: '+str(regressor_new.score(X_test,Y_test)))

# Prediting test set results
y_pred_new = regressor.predict(X_test)

# Visualizing the results
x_surf, y_surf = np.meshgrid(np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 10),
                             np.linspace(X_train[:, 2].min(), X_train[:, 2].max(), 10))
z_surf = np.asarray(np.meshgrid(np.linspace(Y_train.min(), Y_train.max(), 100)))
onlyX = pd.DataFrame({'R&D Spend': x_surf.ravel(), 'Marketing': y_surf.ravel()})
onlyY = pd.DataFrame({'Profit': z_surf.ravel()})
surf_regressor = LinearRegression()
surf_regressor.fit(onlyX, onlyY)
fittedY=surf_regressor.predict(onlyX)

from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 1], X_train[:, 2], Y_train, c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('R&D Spends')
ax.set_ylabel('Marketing Spends')
ax.set_zlabel('Profit')
plt.show() 