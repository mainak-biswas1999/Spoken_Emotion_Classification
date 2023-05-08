import numpy as np
import pandas as pd
import os
def featNorm(X, mupath, sigmapath):
    epsilon = 1e-7;
    
    MU = np.sum(np.sum(X, axis=0, keepdims=False), axis=0, keepdims=False)/(X.shape[0]*X.shape[1])
    
    E_X2 = np.sum(np.sum(X*X, axis=0, keepdims=False), axis=0, keepdims=False)/(X.shape[0]*X.shape[1]) 
    SIGMA = np.sqrt(E_X2 - MU*MU)
    
    X_norm = (X - MU)/(SIGMA+epsilon) 

    pd.DataFrame(MU).to_csv(mupath)
    pd.DataFrame(SIGMA).to_csv(sigmapath)
    
    return X_norm;

