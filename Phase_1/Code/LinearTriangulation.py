import numpy as np
from utils import *


# Given the rotation matrix R and camera pose t, and the 3D points P1 and P2 in homogeneous coordinates,
# check if the points satisfy the Triangulation Cheirality Condition
def triangulate(K, R1, C1, R2, C2, x1, x2):
   
    # Compute the projection matrices P1 and P2
    # P1 = np.dot(K, np.dot(R1, np.hstack((np.eye(3), -C1.reshape(3, 1)))))
    # return np.dot(K, np.hstack((R, -np.dot(R, C.reshape(3, 1)))))
    # P2 = np.dot(K, np.dot(R2, np.hstack((np.eye(3), -C2.reshape(3, 1)))))

    P1 = get_projection_matrix(R1, C1, K)
    P2 = get_projection_matrix(R2, C2, K)

    # print(x1[0].shape)
    # print(x1[1].shape)
    # print(x2.shape)
    # print(P1[2].shape)
    # print(P2[2].shape)
    # # Construct the A matrix
    # A = np.vstack((x1[0] * P1[2] - P1[0],
    #                x1[1] * P1[2] - P1[1],
    #                x2[0] * P2[2] - P2[0],
    #                x2[1] * P2[2] - P2[1]))




    # # Compute the SVD of the A matrix
    # _, _, V = np.linalg.svd(A)

    # # Extract the 3D point from the last column of V
    # X = V[-1, :4]

    # # Normalize the homogeneous coordinate
    # X /= X[3]

    
    # return X[:3]

    # I = np.identity(3)
    # C1 = np.reshape(C1, (3, 1))
    # C2 = np.reshape(C2, (3, 1))

    # P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    # P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)

    p_dash_1T = P2[0,:].reshape(1,4)
    p_dash_2T = P2[1,:].reshape(1,4)
    p_dash_3T = P2[2,:].reshape(1,4)

    # A = []
    # A.append((x1[1]*p3T) - p2T)
    # A.append(p1T - (x1[0]*p3T))
    # A.append((x2[1]*p_dash_3T) - p_dash_2T)
    # A.append(p_dash_1T - (x2[0]*p_dash_3T))

    # A = np.array(A).reshape(4,4)

    # _, _, V = np.linalg.svd(A)
    # X = V[-1, :4]
    # X /= X[3]

    # return X[:3]

    x_list = []
    for i in range(x1.shape[0]):
        x = x1[i][0]
        y = x1[i][1]
        x_dash = x2[i][0]
        y_dash = x2[i][1]


        A = []
        A.append((y * p3T) -  p2T)
        A.append(p1T -  (x * p3T))
        A.append((y_dash * p_dash_3T) -  p_dash_2T)
        A.append(p_dash_1T -  (x_dash * p_dash_3T))

        A = np.array(A).reshape(4,4)

        _, _, V = np.linalg.svd(A)

        V = V.T
        x = V[:,-1]
        x_list.append(x)
    return np.array(x_list)







    



