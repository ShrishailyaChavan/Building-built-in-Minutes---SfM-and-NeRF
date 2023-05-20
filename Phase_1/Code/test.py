import numpy as np

# def get_projection_matrix(R, C, K):
#     return np.dot(K, np.hstack((R, -np.dot(R, C))))  # No need to reshape C inside this function


# def main():
#     K = np.array([[531.122155322710, 0, 407.192550839899],
#                   [0, 531.541737503901, 313.308715048366],
#                   [0, 0, 1]])
#     R1 = np.identity(3)
#     C1 = np.zeros((3, 1))  # C1 is now a 3x1 column vector
#     C1 = C1.reshape(3,)  # Reshape C1 back to a 1D array of length 3
#     P1 = get_projection_matrix(R1, C1, K)


def get_projection_matrix(R, C, K):
    return np.dot(K, np.hstack((R, -np.dot(R, C.reshape(3, 1)))))                                                                            
    
def main():
    K = np.array([[531.122155322710, 0 ,407.192550839899],[0, 531.541737503901, 313.308715048366],[0,0,1]])
    R1 = np.identity(3)
    C1 = np.zeros((3,1))
    P1 = get_projection_matrix(R1, C1, K) 