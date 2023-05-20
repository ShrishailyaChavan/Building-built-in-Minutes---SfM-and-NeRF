import numpy as np
from utils import * 

def disambiguate_pose(R_list, C_list, X_list):
    best_i = 0
    max_positive_depths = 0
    
    for i in range(len(R_list)):
        R, C = R_list[i], C_list[i].reshape(-1,1) 
        r_3D = R[2, :].reshape(1,-1)
        X = X_list[i]
        X = X / X[:,3].reshape(-1,1)
        X = X[:, 0:3]
        n_positive_depths = check_cheirality(X, r_3D,C)
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths
#         print(n_positive_depths, i, best_i)

    R, C, X = R_list[best_i], C_list[best_i], X_list[best_i]

    return R, C, X

def check_cheirality(X_3D, R, C):
    # R(X-C) alone doesnt solve the check positivity. z = X[2] must also be +ve 
    n_positive_depths=  0
    for X in X_3D:
        X = X.reshape(-1,1) 
        if R.dot(X-C)>0 and X[2]>0: 
            n_positive_depths+=1
    return n_positive_depths
