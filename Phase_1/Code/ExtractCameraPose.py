import numpy as np
from scipy.linalg import svd

def decompose_essential_matrix(E):
    """
    Decompose the essential matrix into the relative rotation and translation between two cameras.

    Args:
        E (numpy.ndarray): Essential matrix (3x3).

        K (numpy.ndarray): Camera matrix (3x3).

    Returns:
        numpy.ndarray: Relative rotation (3x3).
        numpy.ndarray: Camera Centers.
    """

    # SVD of the essential matrix
    U, S, Vt = svd(E)

    R = []
    C = []

    
    C1 = U[:, 2]
    C.append(C1)
    C2 = -U[:,2]
    C.append(C2)
    C3 = U[:,2]
    C.append(C3)
    C4 = -U[:,2]
    C.append(C4)
    
    # Check the determinant of U and Vt
    # if np.linalg.det(U) < 0:
    #     U *= -1
    # if np.linalg.det(Vt) < 0:
    #     Vt *= -1

    # Define the matrices for the two possible solutions
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = np.dot(U,np.dot(W,Vt))
    R.append(R1)

    R2 = np.dot(U,np.dot(W,Vt))
    R.append(R2)

    R3 = np.dot(U,np.dot(W,Vt))
    R.append(R3)

    R4 = np.dot(U,np.dot(W,Vt))
    R.append(R4)


    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C
