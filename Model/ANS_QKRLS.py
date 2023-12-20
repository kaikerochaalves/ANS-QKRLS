# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np

class ANS_QKRLS:
    def __init__(self, nu = 0.01, sigma = 0.5, epsilon = 0.01, mu = 1, zeta = 1e-6):
        # Initialize the model's parameters
        self.parameters = pd.DataFrame(columns = ['K', 'Kinv', 'alpha', 'P', 'm', 'Dict'])
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        # Hyperparameters and parameters
        self.sigma = sigma
        # Threshold of ALD sparse rule
        self.nu = nu
        # Threshold for the distance
        self.epsilon = epsilon
        # Threshold for the coherence coefficient
        self.mu = mu
        # Avoid the case in which denominator of the consequent parameters updating becomes zero
        self.zeta = zeta
         
    def fit(self, X, y):

        # Compute the number of samples
        n = X.shape[0]
        
        # Initialize the first input-output pair
        x0 = X[0,].reshape(-1,1)
        y0 = y[0]
        
        # Initialize ANS_QKRLS
        self.Initialize_ANS_QKRLS(x0, y0)

        for k in range(1, n):

            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
                      
            # Update ANS_QKRLS
            k_til = self.ANS_QKRLS(x, y[k])
            
            # Compute output
            Output = self.parameters.loc[0, 'alpha'].T @ k_til
            
            # Store results
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output )
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(y[k]) - Output )
        return self.OutputTrainingPhase
            
    def predict(self, X):

        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T

            # Compute k
            k_til = np.array(())
            for ni in range(self.parameters.loc[0, 'Dict'].shape[1]):
                k_til = np.append(k_til, [self.Kernel(self.parameters.loc[0, 'Dict'][:,ni].reshape(-1,1), x)])
            k_til = k_til.reshape(k_til.shape[0],1)
            
            # Compute the output
            Output = self.parameters.loc[0, 'alpha'].T @ k_til
            
            # Store the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output )

        return self.OutputTestPhase

    def Kernel(self, x1, x2):
        k = np.exp( - ( 1/2 ) * ( (np.linalg.norm( x1 - x2 ))**2 ) / ( self.sigma**2 ) )
        return k
    
    def Initialize_ANS_QKRLS(self, x, y):
        k11 = self.Kernel(x, x)
        K = np.ones((1,1)) * ( k11 )
        Kinv = np.ones((1,1)) / ( k11 )
        alpha = np.ones((1,1)) * y / k11
        NewRow = pd.DataFrame([[K, Kinv, alpha, np.ones((1,1)), 1., x]], columns = ['K', 'Kinv', 'alpha', 'P', 'm', 'Dict'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
        # Initialize first output and residual
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def ANS_QKRLS(self, x, y):
        i = 0
        # Compute k
        k = np.array(())
        for ni in range(self.parameters.loc[i, 'Dict'].shape[1]):
            k = np.append(k, [self.Kernel(self.parameters.loc[i, 'Dict'][:,ni].reshape(-1,1), x)])
        k_til = k.reshape(-1,1)
        # Compute a
        a = np.matmul(self.parameters.loc[i, 'Kinv'], k)
        A = a.reshape(-1,1)
        delta = self.Kernel(x, x) - ( k_til.T @ A ).item()
        if delta == 0:
            delta = 1.
        # Compute the maximum value of the kernel
        mu = max(np.abs(k_til))
        # Compute the distances between x and the Dictionary
        distance = []
        for ni in range(self.parameters.loc[i, 'Dict'].shape[1]):
            distance.append(np.linalg.norm(self.parameters.loc[i, 'Dict'][:,ni].reshape(-1,1) - x))
        # Find the index of minimum distance
        j = np.argmin(distance)
        # Estimating the error
        EstimatedError = ( y - np.matmul(k_til.T, self.parameters.loc[i, 'alpha']) ).item()
        # Novelty criterion
        if delta > self.nu and mu.item() <= self.mu and distance[j] > self.epsilon:
            self.parameters.at[i, 'Dict'] = np.hstack([self.parameters.loc[i, 'Dict'], x])
            self.parameters.at[i, 'm'] = self.parameters.loc[i, 'm'] + 1
            # Update K  
            ktt = self.Kernel(x, x).reshape(1,1)                    
            self.parameters.at[i, 'K'] = np.lib.pad(self.parameters.loc[i,  'K'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeK = self.parameters.loc[i,  'K'].shape[0] - 1
            self.parameters.at[i, 'K'][sizeK,sizeK] = ktt
            self.parameters.at[i, 'K'][0:sizeK,sizeK] = k_til.flatten()
            self.parameters.at[i, 'K'][sizeK,0:sizeK] = k_til.flatten()
            # Updating Kinv                      
            self.parameters.at[i, 'Kinv'] = (1/delta)*(self.parameters.loc[i, 'Kinv'] * delta + np.matmul(A, A.T))
            self.parameters.at[i, 'Kinv'] = np.lib.pad(self.parameters.loc[i, 'Kinv'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeKinv = self.parameters.loc[i,  'Kinv'].shape[0] - 1
            self.parameters.at[i, 'Kinv'][sizeKinv,sizeKinv] = (1/delta)
            self.parameters.at[i, 'Kinv'][0:sizeKinv,sizeKinv] = (1/delta)*(-a)
            self.parameters.at[i, 'Kinv'][sizeKinv,0:sizeKinv] = (1/delta)*(-a)
            # Updating P
            self.parameters.at[i, 'P'] = np.lib.pad(self.parameters.loc[i, 'P'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeP = self.parameters.loc[i,  'P'].shape[0] - 1
            self.parameters.at[i, 'P'][sizeP,sizeP] = 1.
            # Updating alpha
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i, 'alpha'] - ( ( A / delta ) * EstimatedError )
            self.parameters.at[i, 'alpha'] = np.vstack([self.parameters.loc[i, 'alpha'], ( 1 / delta ) * EstimatedError ])
            k_til = np.append(k_til, ktt, axis=0)
            
        elif ( delta <= self.nu or mu.item() > self.mu ) and distance[j] <= self.epsilon:
            # Compute Kinvj and Kj
            Kinvj = self.parameters.loc[i, 'P'][:,j].reshape(-1,1)
            Kj = self.parameters.loc[i, 'K'][:,j].reshape(-1,1)
            # Updating Kinv - Kinv is equivalent to P in QKRLS
            self.parameters.at[i, 'Kinv'] = self.parameters.loc[i, 'Kinv'] - ( Kinvj @ (Kj.T @ self.parameters.loc[i, 'Kinv'] ) ) / ( 1 + Kj.T @ Kinvj )
            # Updating alpha
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i, 'alpha'] + Kinvj @ ( y - Kj.T @ self.parameters.loc[i, 'alpha'] ) / ( ( 1 + Kj.T @ Kinvj ) )
            
        else:
            # Calculating q
            q = np.matmul( self.parameters.loc[i,  'P'], A) / ( 1 + np.matmul(np.matmul(A.T, self.parameters.loc[i, 'P']), A ) )
            # Updating P
            self.parameters.at[i, 'P'] = self.parameters.loc[i, 'P'] - (np.matmul(np.matmul(np.matmul(self.parameters.loc[i, 'P'], A), A.T), self.parameters.loc[i, 'P'])) / ( 1 + np.matmul(np.matmul(A.T, self.parameters.loc[i, 'P']), A))
            # Updating alpha
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i, 'alpha'] + ( np.matmul(self.parameters.loc[i, 'Kinv'], q) * EstimatedError ) / ( self.zeta + np.linalg.norm(k_til)**2 )
        return k_til