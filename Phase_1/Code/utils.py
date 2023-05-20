import numpy as np
from scipy.spatial.transform import Rotation
import glob

def get_projection_matrix(R, C, K):
    return np.dot(K, np.hstack((R, -np.dot(R, C.reshape(3, 1)))))


def h_transform(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def get_rotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def get_quaternion(R2):
    Q = Rotation.from_matrix(R2)
    return Q.as_quat()

def get_euler(R2):
    euler = Rotation.from_matrix(R2)
    return euler.as_rotvec()

def image_count(data_path):
    image_count = len(glob.glob(data_path + "*.png"))

    return image_count

def camera_matrix(data_path):

    filename = data_path + 'calibration.txt'
    # Initialize a 3x3 matrix
    matrix = np.zeros((3, 3))

    # Read the contents of the file and fill the matrix
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            values = line.strip().split()
            for j, value in enumerate(values):
                matrix[i][j] = float(value)

    # Return the matrix
    return matrix


def extract_features(data_path):
    num_of_imgs = image_count(data_path)

    feature_rgb = []
    feature_x = []
    feature_y = []
    feature_flag = []

    for i in range(1, num_of_imgs):
        matching_fname = data_path + "matching" + str(i) + ".txt"
        fname = open(matching_fname, "r")
        nFeatures = 0
        for j, row in enumerate(fname):
            if j == 0:
                row_elems = row.split(':')
                nFeatures = int(row_elems[1])
            else:
                row_x = np.zeros((1, num_of_imgs))
                row_y = np.zeros((1, num_of_imgs))
                row_flag = np.zeros((1, num_of_imgs), dtype=int)
                row_elems = row.split()
                features = [float(x) for x in row_elems]
                features = np.array(features)

                num_of_matches = features[0]
                red = features[1]
                green = features[2]
                blue = features[3]

                feature_rgb.append([red, green, blue])

                src_x = features[4]
                src_y = features[5]

                row_x[0, i-1] = src_x
                row_y[0, i-1] = src_y
                row_flag[0, i-1] = 1

                j = 1
                while num_of_matches > 1:
                    img_id = int(features[5+j])
                    img_id_x = features[6+j]
                    img_id_y = features[7+j]
                    j = j+3
                    num_of_matches = num_of_matches - 1

                    row_x[0, img_id - 1] = img_id_x
                    row_y[0, img_id - 1] = img_id_y
                    row_flag[0, img_id - 1] = 1

                feature_x.append(row_x)
                feature_y.append(row_y)
                feature_flag.append(row_flag)

    return np.array(feature_x).reshape(-1, num_of_imgs), \
        np.array(feature_y).reshape(-1, num_of_imgs), np.array(feature_flag).reshape(-1, num_of_imgs), \
            np.array(feature_rgb).reshape(-1, 3)

def draw_features():
    pass

def reprojection_error(X, p1, p2, R1, C1, R2, C2, K):
    P1 = get_projection_matrix(R1, C1, K)
    P2 = get_projection_matrix(R2, C2, K)

    p1_1t, p1_2t, p1_3t = P1
    p1_1t, p1_2t, p1_3t = p1_1t.reshape(1,4), p1_2t.reshape(1,4), p1_3t.reshape(1,4)
    p2_1t, p2_2t, p2_3t = P2
    p2_1t, p2_2t, p2_3t = p2_1t.reshape(1,4), p2_2t.reshape(1,4), p2_3t.reshape(1,4)

    X = X.reshape(4,1)

    # Reprojection error w.r.t camera 1 ref
    u1, v1 = p1[0], p1[1]

    u1_proj = np.divide(p1_1t.dot(X), p1_3t.dot(X))
    v1_proj = np.divide(p1_2t.dot(X), p1_3t.dot(X))

    error1 =  np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

    # Reprojection error w.r.t camera 2 ref
    u2, v2 = p2[0], p2[1]

    u2_proj = np.divide(p2_1t.dot(X), p2_3t.dot(X))
    v2_proj = np.divide(p2_2t.dot(X), p2_3t.dot(X))

    error2 = np.square(v2 - v2_proj) + np.square(u2 - u2_proj)

    return error1, error2


