# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 02:05:45 2022

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""

# Importing libraries
import math
import numpy as np
import statistics as st
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt 

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Including to the path another fold
import sys

# Including to the path another fold
sys.path.append(r'Model')
sys.path.append(r'Functions')

# Importing the model
from ANS_QKRLS import ANS_QKRLS

# Importing the library to generate the time series
from LorenzAttractorGenerator import Lorenz


#-----------------------------------------------------------------------------
# Generating the Lorenz Attractor time series
#-----------------------------------------------------------------------------

Serie = "Lorenz"

# Input parameters
x0 = 0.
y0 = 1.
z0 = 1.05
sigma = 10
beta = 2.667
rho=28
num_steps = 10000

# Creating the Lorenz Time Series
x, y, z = Lorenz(x0 = x0, y0 = y0, z0 = z0, sigma = sigma, beta = beta, rho = rho, num_steps = num_steps)

# Ploting the graphic

ax = plt.figure(figsize=(19.20,10.80)).add_subplot(projection='3d')
ax.plot(x, y, z, lw = 0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

def Create_Leg(data, ncols, leg, leg_output = None):
    X = np.array(data[leg*(ncols-1):].reshape(-1,1))
    for i in range(ncols-2,-1,-1):
        X = np.append(X, data[leg*i:leg*i+X.shape[0]].reshape(-1,1), axis = 1)
    X_new = np.array(X[:,-1].reshape(-1,1))
    for col in range(ncols-2,-1,-1):
        X_new = np.append(X_new, X[:,col].reshape(-1,1), axis=1)
    if leg_output == None:
        return X_new
    else:
        y = np.array(data[leg*(ncols-1)+leg_output:].reshape(-1,1))
        return X_new[:y.shape[0],:], y
        

# Defining the atributes and the target value
X = np.concatenate([x[:-1].reshape(-1,1), y[:-1].reshape(-1,1), z[:-1].reshape(-1,1)], axis = 1)
y = x[1:].reshape(-1,1)

# Spliting the data into train and test
X_train, X_test = X[:8000,:], X[8000:,:]
y_train, y_test = y[:8000,:], y[8000:,:]

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, label='Actual Value', color='red')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper right')
plt.show()



# #-----------------------------------------------------------------------------
# # Feature scaling
# #-----------------------------------------------------------------------------


# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
#print(scaler.data_max_)
X_test = scaler.transform(X_test)


#-----------------------------------------------------------------------------
# Calling the model
#-----------------------------------------------------------------------------

Model = "ANS-QKRLS"

# Setting the hyperparameters
nu = 0.01
sigma = 0.5
epsilon = 0.01
mu = 1
zeta = 1e-6

# Initializing the model
model = ANS_QKRLS(nu = nu, sigma = sigma, epsilon = epsilon, mu = mu, zeta = zeta)
# Train the model
OutputTraining = model.fit(X_train, y_train)
# Test the model
OutputTest = model.predict(X_test)

#-----------------------------------------------------------------------------
# Evaluate the model's performance
#-----------------------------------------------------------------------------

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, OutputTest))
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, OutputTest)

# Printing the RMSE
print("RMSE = ", RMSE)
# Printing the NDEI
print("NDEI = ", NDEI)
# Printing the MAE
print("MAE = ", MAE)


#-----------------------------------------------------------------------------
# Plot the graphics
#-----------------------------------------------------------------------------

# Plot the graphic
plt.figure(figsize=(19.20,10.80))
plt.rc('font', size=30)
plt.rc('axes', titlesize=30)
plt.plot(y_test, label='Actual Value', color='red')
plt.plot(OutputTest, color='blue', label='ANS-QKRLS')
plt.ylabel('Output')
plt.xlabel('Samples')
plt.legend(loc='upper right')
plt.savefig(f'Graphics/{Model}_{Serie}.eps', format='eps', dpi=1200)
plt.show()
