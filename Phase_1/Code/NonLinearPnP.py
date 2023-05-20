import numpy as np
from scipy.spatial.transform import Rotation
import scipy.optimize as optimize
from NonLinearTriangulation import *
from LinearPnP import *
from utils import *


def PnPLoss(X0, x3D, pts, K):
    
    Q, C = X0[:4], X0[4:].reshape(-1,1)
    R = get_rotation(Q)
    P = get_projection_matrix(R,C,K)
    
    E = []
    for X, pt in zip(x3D, pts):

        p_1T, p_2T, p_3T = P# rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)


        X = h_transform(X.reshape(1,-1)).reshape(-1,1) 
        ## reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        e = np.square(v - v_proj) + np.square(u - u_proj)

        E.append(e)

    sumError = np.mean(np.array(E).squeeze())
    return sumError

def NonLinearPnP(K, pts, x3D, R0, C0):
    """    
    K : Camera Matrix
    pts1, pts2 : 2D point correspondences
    x_init_3d :  initial 3D point 
    R2, C2 : relative camera pose - estimated from PnP
    Returns:
        x_optim_3d : optimized 3D points
    """

    Q = get_quaternion(R0)
    X0 = [Q[0] ,Q[1],Q[2],Q[3], C0[0], C0[1], C0[2]] 

    optimized_params = optimize.least_squares(
        fun = PnPLoss,
        x0=X0,
        method="trf",
        args=[x3D, pts, K])
    X1 = optimized_params.x
    Q = X1[:4]
    C = X1[4:]
    R = get_rotation(Q)
    return R, C

