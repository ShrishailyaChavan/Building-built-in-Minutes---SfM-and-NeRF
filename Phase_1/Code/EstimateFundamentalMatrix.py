import numpy as np
import cv2

def get_fundamental_matrix(pts1, pts2):
    # Normalize the points to improve numerical stability
    pts1_norm = cv2.normalize(pts1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    pts2_norm = cv2.normalize(pts2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Build the A matrix
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Find the singular value decomposition of A
    _, _, v = np.linalg.svd(A)
    
    # The fundamental matrix is the last column of V
    F = v[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint by performing SVD and setting smallest singular value to 0
    u, s, vh = np.linalg.svd(F)
    s[-1] = 0
    S = np.diag(s)
    F = np.dot(np.dot(u, S), vh)
    
    # Denormalize the fundamental matrix
    T = np.array([[1/np.amax(pts1[:,0]), 0, 0], 
                  [0, 1/np.amax(pts1[:,1]), 0], 
                  [0, 0, 1]])
    F = np.dot(np.dot(T.T, F), T)
    
    return F


def get_epipoles(F):
    # Find the left and right singular vectors of F
    _, _, v = np.linalg.svd(F)
    e1 = v[-1]
    
    _, _, v = np.linalg.svd(F.T)
    e2 = v[-1]
    
    # Normalize the epipoles
    e1 /= e1[2]
    e2 /= e2[2]
    
    return e1, e2

def epipolars_line(F, pts1, pts2):
    # Compute epipolar line in image 2 from v1
    l2 = np.dot(F, pts1)
    l1 = np.dot(F.T, pts2)

     # Normalize the line
    l2_norm = np.sqrt(l2[0]**2 + l2[1]**2)
    l2 /= l2_norm

    l1_norm = np.sqrt(l1[0]**2 + l1[1]**2)
    l1 /= l1_norm

    return l1, l2






