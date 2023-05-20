import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from BuildVisibilityMatrix import *
from utils import *

def get_2d_pts(X_idx, visiblity_matrix, feature_x, feature_y):
    "Get 2D Points from the feature x and feature y having same index from the visibility matrix"
    pts_2d = []
    visible_feature_x = feature_x[X_idx]
    visible_feature_y = feature_y[X_idx]
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                pt = np.hstack((visible_feature_x[i,j], visible_feature_y[i,j]))
                pts_2d.append(pt)
    return np.array(pts_2d).reshape(-1, 2) 

def get_cam_pt_indices(visiblity_matrix):

    "From Visibility Matrix take away indices of point visible from Camera pose by taking indices of cam as well"

    cam_indices = []
    pt_indices = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                cam_indices.append(j)
                pt_indices.append(i)

    return np.array(cam_indices).reshape(-1), np.array(pt_indices).reshape(-1)

def bundle_adjustment_sparsity(X_found, filtered_feature_flag, nCam):
    
    "Create Sparsity Matrix"

    number_of_cam = nCam + 1
    X_idx, visiblity_matrix = get_obs_index_and_viz_mat(X_found.reshape(-1), filtered_feature_flag, nCam)
    n_observations = np.sum(visiblity_matrix)
    n_points = len(X_idx[0])

    m = n_observations * 2
    n = number_of_cam * 6 + n_points * 3   #Here we don't take focal length and 2 radial distortion parameters, i.e no - 6. We just refine orientation and translation of 3d point and not cam parameters.
    A = lil_matrix((m, n), dtype=int)
    # print(m, n)


    i = np.arange(n_observations)
    camera_indices, point_indices = get_cam_pt_indices(visiblity_matrix)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, (nCam)* 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, (nCam) * 6 + point_indices * 3 + s] = 1

    return A    

def project(points_3d, camera_params, K):
    def project_pt_(R, C, pt_3d, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        x_3d_4 = np.hstack((pt_3d, 1))
        x_proj = np.dot(P2, x_3d_4.T)
        x_proj /= x_proj[-1]
        return x_proj

    x_proj = []
    for i in range(len(camera_params)):
        R = get_rotation(camera_params[i, :3], 'e')
        C = camera_params[i, 3:].reshape(3,1)
        pt_3d = points_3d[i]
        pt_proj = project_pt_(R, C, pt_3d, K)[:2]
        x_proj.append(pt_proj)    
    return np.array(x_proj)

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def fun(x0, nCam, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    number_of_cam = nCam + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()
    
    return error_vec  


def bundle_adjustment(X_idx,visibility_matrix,X_all,X_found,feature_x,feature_y, filtered_feature_flag, R_set, C_set, K, nCam):
    points_3d = X_all[X_idx]
    points_2d = get_2d_pts(X_idx,visibility_matrix,feature_x,feature_y)

    RC = []
    for i in range(nCam+1):
        C, R = C_set[i], R_set[i]
        Q = get_euler(R)
        RC_ = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC.append(RC_)
    RC = np.array(RC, dtype=object).reshape(-1,6)

    x0 = np.hstack((RC.ravel(), points_3d.ravel()))
    n_pts = points_3d.shape[0]

    camera_indices, points_indices = get_cam_pt_indices(visibility_matrix)

    A = bundle_adjustment_sparsity(X_found,filtered_feature_flag,nCam)
    t0 = time.time()
    res = least_squares(fun,x0,jac_sparsity=A, verbose=2,x_scale='jac', ftol=1e-10, method='trf',
                        args=(nCam, n_pts, camera_indices, points_indices, points_2d,K))

    t1 = time.time()
    print("Time required to run Bundle Adj: ", t1-t0, "s \nA matrix shape: ",A.shape,"\n######")

    x1 = res.x
    no_of_cams = nCam + 1
    cam_param_optim = x1[:no_of_cams*6].reshape((no_of_cams,6))
    pts_3d_optim = x1[no_of_cams*6:].reshape((n_pts,3))

    X_all_optim = np.zeros_like(X_all)
    X_all_optim[X_idx] = pts_3d_optim

    C_optim , R_optim = [], []
    for i in range(len(cam_param_optim)):
        R = get_rotation(cam_param_optim[i,:3], 'e')
        C = cam_param_optim[i,3:].reshape(3,1)
        C_optim.append(C)
        R_optim.append(R)

    return R_optim, C_optim, X_all_optim