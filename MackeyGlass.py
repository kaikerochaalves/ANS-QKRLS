# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:00:12 2022

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
from MackeyGlassGenerator import MackeyGlass


#-----------------------------------------------------------------------------
# Generating the Mackey-Glass time series
#-----------------------------------------------------------------------------

Serie = "MackeyGlass"

# The theory
# Mackey-Glass time series refers to the following, delayed differential equation:
    
# dx(t)/dt = ax(t-\tau)/(1 + x(t-\tau)^10) - bx(t)


# Input parameters
a        = 0.2;     # value for a in eq (1)
b        = 0.1;     # value for b in eq (1)
tau      = 17;		# delay constant in eq (1)
x0       = 1.2;		# initial condition: x(t=0)=x0
sample_n = 6000;	# total no. of samples, excluding the given initial condition

# MG = mackey_glass(N, a = a, b = b, c = c, d = d, e = e, initial = initial)
MG = MackeyGlass(a = a, b = b, tau = tau, x0 = x0, sample_n = sample_n)

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
X, y = Create_Leg(MG, ncols = 4, leg = 6, leg_output = 85)

# Spliting the data into train and test
X_train, X_test = X[201:3201,:], X[5001:5501,:]
y_train, y_test = y[201:3201,:], y[5001:5501,:]


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
nu = 0.005
sigma = 0.1
epsilon = 0.01
mu = 0.98
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