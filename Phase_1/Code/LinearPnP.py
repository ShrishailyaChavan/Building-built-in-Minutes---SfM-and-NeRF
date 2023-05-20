import numpy as np
from NonLinearTriangulation import *
from utils import *


def PnP_reprojection_error(x_3d, pts, K, R, C):
    P = get_projection_matrix(R,C,K)

    E = []
    for X, pt in zip(x_3d, pts):
        p_1T, p_2T, p_3T = P# rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
        X = h_transform(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
        ## reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        e = np.square(v - v_proj) + np.square(u - u_proj)

        E.append(e)

    mean_err = np.mean(np.array(E).squeeze())
    return mean_err


def LinearPnP(X, x, K):
    N = X.shape[0]
    
    X_ht = h_transform(X)
    x_ht = h_transform(x)
    
    # normalize x
    K_inv = np.linalg.inv(K)
    x_n = K_inv.dot(x_ht.T).T
    
    for i in range(N):
        X_i = X_ht[i].reshape((1, 4))
        zeros = np.zeros((1, 4))
        
        u, v, _ = x_n[i]
        
        u_cross = np.array([[0, -1, v],
                            [1,  0 , -u],
                            [-v, u, 0]])
        X_tilde = np.vstack((np.hstack((X_i, zeros, zeros)), 
                            np.hstack((zeros,     X_i, zeros)), 
                            np.hstack((zeros, zeros,     X_i))))
        a = u_cross.dot(X_tilde)
        
        if i > 0:
            A = np.vstack((A, a))
        else:
            A = a
            
    _, _, V = np.linalg.svd(A)
    P = V[-1].reshape((3, 4))
    R = P[:, :3]
    U_r, D, V_r = np.linalg.svd(R) # to enforce Orthonormality
    R = U_r.dot(V_r)
    
    C = P[:, 3]
    C = - np.linalg.inv(R).dot(C)
    
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
        
    return R, C


