import numpy as np


def get_obs_index_and_viz_mat(X_found, filtered_feature_flag, nCam):
    # find the 3d points such that they are visible in either of the cameras < nCam
    bin_temp = np.zeros((filtered_feature_flag.shape[0]), dtype = int)
    for n in range(nCam + 1):
        bin_temp = bin_temp | filtered_feature_flag[:,n]

    X_idx = np.where((X_found.reshape(-1)) & (bin_temp))
    
    visiblity_matrix = X_found[X_idx].reshape(-1,1)
    for n in range(nCam + 1):
        visiblity_matrix = np.hstack((visiblity_matrix, filtered_feature_flag[X_idx, n].reshape(-1,1)))

    _, c = visiblity_matrix.shape
    return X_idx, visiblity_matrix[:, 1:c]