import numpy as np
from numpy.linalg import svd
import random

from EstimateFundamentalMatrix import *


random.seed(19)

"""
def find_fundamental_matrix(x1, x2):
    # Normalize the points to improve the stability of the computation
    x1_norm = (x1 - np.mean(x1, axis=0)) / np.std(x1, axis=0)
    x2_norm = (x2 - np.mean(x2, axis=0)) / np.std(x2, axis=0)

    # Construct the matrix A for the fundamental matrix
    A = np.hstack((x1_norm, np.ones((x1.shape[0], 1))))
    B = np.hstack((x2_norm, np.ones((x2.shape[0], 1))))
    AB = np.einsum('ij,ik->ijk', A, B)
    AB = AB.reshape((-1, AB.shape[-1]))

    # Solve for the fundamental matrix using SVD
    _, _, V = svd(AB)
    F = V[-1, :].reshape(3, 3)

    # Enforce the rank-2 constraint on the fundamental matrix
    U, S, V = svd(F)
    S[-1] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    # Denormalize the fundamental matrix
    F = np.dot(np.dot(np.transpose(x2), F), x1)

    return F

"""

def compute_residuals(F, x1, x2):
    # Compute the epipolar line for each point in x1
    l = np.dot(F, np.vstack((x1.T, np.ones((1, x1.shape[0])))))
    d = np.sqrt(l[0,:]**2 + l[1,:]**2)
    l = l / d

    # Compute the distance from each point in x2 to its corresponding epipolar line
    x2_hom = np.vstack((x2.T, np.ones((1, x2.shape[0]))))
    distances = np.abs(np.sum(l * x2_hom, axis=0))

    return distances

def find_inliers(x1, x2, F, threshold):
    # Compute the residuals for each correspondence
    residuals = compute_residuals(F, x1, x2)

    # Identify the inliers based on the residuals
    inliers = residuals < threshold

    return inliers

def ransac_fundamental_matrix(x1, x2, num_iterations, threshold):
    best_F = None
    best_num_inliers = 0
    for i in range(num_iterations):
        # Randomly select 8 correspondences
        idx = np.random.choice(x1.shape[0], 8, replace=False)
        F = get_fundamental_matrix(x1[idx, :], x2[idx, :])

        # Compute the number of inliers
        inliers = find_inliers(x1, x2, F, threshold)
        num_inliers = np.sum(inliers)

        # Update the best estimate if necessary
        if num_inliers > best_num_inliers:
            best_F = F
            best_num_inliers = num_inliers

    # Refit the model using all of the inliers
    inliers = find_inliers(x1, x2, best_F, threshold)
    best_F = get_fundamental_matrix(x1[inliers, :], x2[inliers, :])

    return best_F, inliers
