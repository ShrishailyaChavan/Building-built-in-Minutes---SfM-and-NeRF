import numpy as np
import cv2
import scipy.optimize as optimize
from utils import *


def reprojection_loss(X, pts1, pts2, projection_matrix1, projection_matrix2):
    
    p1_1T, p1_2T, p1_3T = projection_matrix1 # rows of projection_matrix1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

    p2_1T, p2_2T, p2_3T = projection_matrix2 # rows of projection_matrix2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

    ## reprojection error for reference camera points - j = 1
    u1,v1 = pts1[0], pts1[1]
    u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
    v1_proj =  np.divide(p1_2T.dot(X) , p1_3T.dot(X))
    E1 = np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

    
    ## reprojection error for second camera points - j = 2    
    u2,v2 = pts2[0], pts2[1]
    u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
    v2_proj =  np.divide(p2_2T.dot(X) , p2_3T.dot(X))    
    E2 = np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
    
    error = E1 + E2
    return error.squeeze()   
    
def non_linear_triangulation(K, x1, x2, X_init_3d, R1, C1, R2, C2):
    
    P1 = get_projection_matrix(R1, C1, K) 
    P2 = get_projection_matrix(R2, C2, K)
    
    assert x1.shape[0] == x2.shape[0] == X_init_3d.shape[0], 'Different shape between 2D to 3d correspondences'

    X_optim_3d = []
    for i in range(len(X_init_3d)):
        optimized_params = optimize.least_squares(fun=reprojection_loss, x0=X_init_3d[i], method="trf", args=[x1[i], x2[i], P1, P2])
        X = optimized_params.x
        X_optim_3d.append(X)
        
    return np.array(X_optim_3d)


