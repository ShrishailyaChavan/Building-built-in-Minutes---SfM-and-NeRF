from LinearPnP import *
from NonLinearTriangulation import *
import numpy as np
from utils import *

def PnPError(x, X, R, C, K):
    u,v = x
    X = h_transform(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
    C = C.reshape(-1, 1)
    P = get_projection_matrix(R,C,K)
    p1, p2, p3 = P
        
    u_proj = np.divide(p1.dot(X) , p3.dot(X))
    v_proj =  np.divide(p2.dot(X) , p3.dot(X))

    x_proj = np.hstack((u_proj, v_proj))
    x = np.hstack((u, v))
    e = np.linalg.norm(x - x_proj)
    return  e

def PnPRANSAC(K, features, x_3d, n_iterations = 1000, error_thresh = 5):

    inliers_thresh = 0
    # chosen_indices = []
    best_R, best_t = None, None
    n_rows = x_3d.shape[0]
    
    for i in range(0, n_iterations):
        
        #select 6 points randomly
        random_indices = np.random.choice(n_rows, size=6)
        X, x = x_3d[random_indices], features[random_indices]
        
        R,C = LinearPnP(X, x, K)
        
        indices = []
        if R is not None:
            for j in range(n_rows):
                feature = features[j]
                X = x_3d[j]
                error = PnPError(feature, X, R, C, K)

                if error < error_thresh:
                    indices.append(j)
                    
        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            best_R = R
            best_t = C
            
    return best_R, best_t