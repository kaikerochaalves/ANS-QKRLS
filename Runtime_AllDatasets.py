# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:46:49 2023

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import numpy as np
import pandas as pd
import statistics as st
import timeit
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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
from NonlinearGenerator import Nonlinear
from LorenzAttractorGenerator import Lorenz


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

simul = 0
l_time_train = []
l_time_test = []
columns = ['simulation', 'runtime_train', 'runtime_test']
result = pd.DataFrame(columns = columns)
for i in range(30):
    
    # Initializing the model
    model = ANS_QKRLS(nu = nu, sigma = sigma, epsilon = epsilon, mu = mu, zeta = zeta)
    
    # Initial time train
    start = timeit.default_timer()
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Final time train
    end = timeit.default_timer()
    # Runtime train
    runtime_train = end - start
    l_time_train.append(runtime_train)
    
    # Initial time test
    start = timeit.default_timer()
    # Test the model
    OutputTest = model.predict(X_test)
    # Final time test
    end = timeit.default_timer()
    # Runtime test
    runtime_test = end - start
    l_time_test.append(runtime_test)
    
    simul = simul + 1
    print(f'Simulação: {simul}')
    
    NewRow = pd.DataFrame([[simul, runtime_train, runtime_test]], columns = columns)
    result = pd.concat([result, NewRow], ignore_index=True)

# Train
mean_train = st.mean(l_time_train)
std_train = st.stdev(l_time_train)

# Test
mean_test = st.mean(l_time_test)
std_test = st.stdev(l_time_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, OutputTest))
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, OutputTest)

r1 = f'{Serie} & {mean_train:.2f} $\pm$ {std_train:.2f} & {mean_test:.2f} $\pm$ {std_test:.2f} & {RMSE:.2f} & {NDEI:.2f} & {MAE:.2f}'
print("\n", r1, "\n")

NewRow = pd.DataFrame([['Summary', f'{mean_train}|{std_train}', f'{mean_test}|{std_test}']], columns = columns)
result = pd.concat([result, NewRow], ignore_index=True)  

name = f"RuntimeResults\Runtime_{Model}_{Serie}.xlsx"
result.to_excel(name)


#-----------------------------------------------------------------------------
# Generating the Nonlinear time series
#-----------------------------------------------------------------------------

Serie = "Nonlinear"

sample_n = 6000
NTS, u = Nonlinear(sample_n)
        

# Defining the atributes and the target value
X, y = Create_Leg(NTS, ncols = 2, leg = 1, leg_output = 1)
X = np.append(X, u[:X.shape[0]].reshape(-1,1), axis = 1)

# Spliting the data into train and test
X_train, X_test = X[2:5002,:], X[5002:5202,:]
y_train, y_test = y[2:5002,:], y[5002:5202,:]


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
sigma = 0.1
epsilon = 0.01
mu = 1
zeta = 1e-6

simul = 0
l_time_train = []
l_time_test = []
columns = ['simulation', 'runtime_train', 'runtime_test']
result = pd.DataFrame(columns = columns)
for i in range(30):
    
    # Initializing the model
    model = ANS_QKRLS(nu = nu, sigma = sigma, epsilon = epsilon, mu = mu, zeta = zeta)
    
    # Initial time train
    start = timeit.default_timer()
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Final time train
    end = timeit.default_timer()
    # Runtime train
    runtime_train = end - start
    l_time_train.append(runtime_train)
    
    # Initial time test
    start = timeit.default_timer()
    # Test the model
    OutputTest = model.predict(X_test)
    # Final time test
    end = timeit.default_timer()
    # Runtime test
    runtime_test = end - start
    l_time_test.append(runtime_test)
    
    simul = simul + 1
    print(f'Simulação: {simul}')
    
    NewRow = pd.DataFrame([[simul, runtime_train, runtime_test]], columns = columns)
    result = pd.concat([result, NewRow], ignore_index=True)

# Train
mean_train = st.mean(l_time_train)
std_train = st.stdev(l_time_train)

# Test
mean_test = st.mean(l_time_test)
std_test = st.stdev(l_time_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, OutputTest))
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, OutputTest)

r2 = f'{Serie} & {mean_train:.2f} $\pm$ {std_train:.2f} & {mean_test:.2f} $\pm$ {std_test:.2f} & {RMSE:.2f} & {NDEI:.2f} & {MAE:.2f}'
print("\n", r2, "\n")

NewRow = pd.DataFrame([['Summary', f'{mean_train}|{std_train}', f'{mean_test}|{std_test}']], columns = columns)
result = pd.concat([result, NewRow], ignore_index=True)  

name = f"RuntimeResults\Runtime_{Model}_{Serie}.xlsx"
result.to_excel(name)


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

# Defining the atributes and the target value
X = np.concatenate([x[:-1].reshape(-1,1), y[:-1].reshape(-1,1), z[:-1].reshape(-1,1)], axis = 1)
y = x[1:].reshape(-1,1)

# Spliting the data into train and test
X_train, X_test = X[:8000,:], X[8000:,:]
y_train, y_test = y[:8000,:], y[8000:,:]


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

simul = 0
l_time_train = []
l_time_test = []
columns = ['simulation', 'runtime_train', 'runtime_test']
result = pd.DataFrame(columns = columns)
for i in range(30):
    
    # Initializing the model
    model = ANS_QKRLS(nu = nu, sigma = sigma, epsilon = epsilon, mu = mu, zeta = zeta)
    
    # Initial time train
    start = timeit.default_timer()
    # Train the model
    OutputTraining = model.fit(X_train, y_train)
    # Final time train
    end = timeit.default_timer()
    # Runtime train
    runtime_train = end - start
    l_time_train.append(runtime_train)
    
    # Initial time test
    start = timeit.default_timer()
    # Test the model
    OutputTest = model.predict(X_test)
    # Final time test
    end = timeit.default_timer()
    # Runtime test
    runtime_test = end - start
    l_time_test.append(runtime_test)
    
    simul = simul + 1
    print(f'Simulação: {simul}')
    
    NewRow = pd.DataFrame([[simul, runtime_train, runtime_test]], columns = columns)
    result = pd.concat([result, NewRow], ignore_index=True)

# Train
mean_train = st.mean(l_time_train)
std_train = st.stdev(l_time_train)

# Test
mean_test = st.mean(l_time_test)
std_test = st.stdev(l_time_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, OutputTest))
# Compute the Non-Dimensional Error Index
NDEI= RMSE/st.stdev(y_test.flatten())
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, OutputTest)

r3 = f'{Serie} & {mean_train:.2f} $\pm$ {std_train:.2f} & {mean_test:.2f} $\pm$ {std_test:.2f} & {RMSE:.2f} & {NDEI:.2f} & {MAE:.2f}'
print("\n", r3, "\n")

NewRow = pd.DataFrame([['Summary', f'{mean_train}|{std_train}', f'{mean_test}|{std_test}']], columns = columns)
result = pd.concat([result, NewRow], ignore_index=True)  

name = f"RuntimeResults\Runtime_{Model}_{Serie}.xlsx"
result.to_excel(name)

print("Results:")
print("\n", r1, "\n")
print(r2, "\n")
print(r3, "\n")