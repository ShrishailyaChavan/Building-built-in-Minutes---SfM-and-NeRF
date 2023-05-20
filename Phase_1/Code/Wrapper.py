import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse 
import glob
from EstimateFundamentalMatrix import *
from GetInlierRANSANC import * 
from EssentialMatrixFromFundamentalMatrix import * 
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonLinearTriangulation import *
from PnPRansac import *
from LinearPnP import *
from NonLinearPnP import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *
from utils import *


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Outputs', default='/home/jc-merlab/RBE549_P2/Phase_1/Output/', help='Outputs are saved here')
    Parser.add_argument('--Data', default='/home/jc-merlab/P2_Phase1/P3Data/', help='Data')

    Args = Parser.parse_args()
    Data = Args.Data
    Output = Args.Outputs

    num_of_images = image_count(Data)
    print(num_of_images)

    #Images
    images = []
    for i in range(1,num_of_images+1): #5 images given
        path = Data + str(i) + ".png"
        image = cv2.imread(path)
        if image is not None:
            images.append(image)
        else:
            print("No image is found")

    #Feature Correspondence
    "We have 5 images and 4 matching .txt files"
    "nFeatures: (the number of feature points of the ith image - each following row specifies matches across images given a feature location in the ith image.)"
    "Each Row: (the number of matches for the jth feature) (Red Value) (Green Value) (Blue Value) (ucurrent image) (vcurrent image) (image id) (uimage id image) (vimage id image) (image id) (uimage id image) (v_{image id image}) …"

    feature_x, feature_y, feature_flag, feature_rgb = extract_features(Data)
    # print(feature_x.shape, feature_y.shape, feature_flag.shape, feature_rgb_values.shape) We get (3833 feature points total)

    filtered_feature_flag = np.zeros_like(feature_flag) #np.zeros has limit which is solve by zeros_like
    f_matrix = np.empty(shape=(5,5), dtype=object)

    for i in range(0,num_of_images-1): #No of Images = 5
        for j in range(i+1,num_of_images):

            idx = np.where(feature_flag[:,i] & feature_flag[:,j])
            pts1 = np.hstack((feature_x[idx,i].reshape((-1,1)), feature_y[idx,i].reshape((-1,1))))
            pts2 = np.hstack((feature_x[idx,j].reshape((-1,1)), feature_y[idx,j].reshape((-1,1))))
            idx = np.array(idx).reshape(-1)
            
            if len(idx) > 8:
                # print(idx)
                # F_inliers, inliers_idx = ransac_fundamental_matrix(pts1,pts2,num_iterations=2000, threshold=0.002)
                F_inliers, inliers_idx = get_F_inliers(pts1,pts2,idx)
                print("Between Images: ",i,"and",j,"NO of Inliers: ", len(inliers_idx), "/", len(idx) )
                f_matrix[i,j] = F_inliers
                # print(f_matrix)
                # print(filtered_feature_flag)
                filtered_feature_flag[inliers_idx,j] = 1
                filtered_feature_flag[inliers_idx,i] = 1
    
    
    print("######Obtained Feature Points after RANSAC#######")
    print("Starting with 1st 2 images")    
   

    # assume images 1 and 2 are stored in the 'images' list
    img1 = images[0]
    img2 = images[1]

    # get inliers for images 1 and 2
    inliers_idx_1 = np.where(filtered_feature_flag[:,0] == 1)[0]
    inliers_idx_2 = np.where(filtered_feature_flag[:,1] == 1)[0]

    # get feature points for images 1 and 2
    features_1 = np.hstack((feature_x[:,0].reshape((-1,1)), feature_y[:,0].reshape((-1,1))))
    features_2 = np.hstack((feature_x[:,1].reshape((-1,1)), feature_y[:,1].reshape((-1,1))))

    # get inlier feature points for images 1 and 2
    inlier_features_1 = features_1[inliers_idx_1, :]
    inlier_features_2 = features_2[inliers_idx_2, :]

    # concatenate images horizontally
    stacked_img = np.concatenate((img1, img2), axis=1)

    # draw lines between inlier feature points
    for i in range(len(inliers_idx_1)):
        pt1 = tuple(inlier_features_1[i])
        pt2 = tuple(inlier_features_2[i]) + np.array([img1.shape[1], 0])
        cv2.line(stacked_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), thickness=1)


    cv2.imwrite(Output+'inlier_matching.png', stacked_img)

    # #Compute Essential Matrix, Estimate Pose, Triangulate
    F12 = f_matrix[0,1]

    print(F12)
    # #K is given
    K = camera_matrix(Data)
    # K = np.array([[531.122155322710, 0 ,407.192550839899],[0, 531.541737503901, 313.308715048366],[0,0,1]])
    E12 = getEssentialMatrix(K,F12)

    #Estimating the Camera Pose
    R_set, C_set = decompose_essential_matrix(E12)

    idx = np.where(filtered_feature_flag[:,0] & filtered_feature_flag[:,1])
    pts1 = np.hstack((feature_x[idx,0].reshape((-1,1)), feature_y[idx,0].reshape((-1,1))))
    pts2 = np.hstack((feature_x[idx,1].reshape((-1,1)), feature_y[idx,1].reshape((-1,1))))

    R1_ = np.identity(3)
    C1_ = np.zeros((3,1))

    # print(C_set)

    pts_3d_4 = []
    for i in range(len(C_set)):
        x1 = pts1
        x2 = pts2
        # C_set[i] = C_set[i].reshape(3,1)
        # print(C_set[i])
        # print(C_set[i].shape)
        X = triangulate(K, R1_, C1_, R_set[i], C_set[i], x1, x2)

        #Now we get 4 poses, we need to select unique one with maximum positive depth points
        X = X/X[:,3].reshape(-1,1)
        pts_3d_4.append(X)

    R_best, C_best, X = disambiguate_pose(R_set, C_set, pts_3d_4)
    X = X/X[:,3].reshape(-1,1)

    #Non-Linear Triangulation
    print("######Performing Non-Linear Triangulation######")
    X_refined = non_linear_triangulation(K,pts1,pts2,X,R1_,C1_,R_best,C_best)
    # print(X_refined.shape)
    X_refined = X_refined / X_refined[:,3].reshape(-1,1)
    # print(X_refined.shape)

    total_err1 = []
    for pt1, pt2, X_3d in zip(pts1,pts2,X):
        err1, err2 = reprojection_error(X_3d,pt1,pt2,R1_,C1_,R_best,C_best,K)
        total_err1.append(err1+err2)
    
    mean_err1 = np.mean(total_err1)

    total_err2 = []
    for pt1, pt2, X_3d in zip(pts1,pts2,X_refined):
        err1, err2 = reprojection_error(X_3d,pt1,pt2,R1_,C1_,R_best,C_best,K)
        total_err2.append(err1+err2)
    
    mean_err2 = np.mean(total_err2)

    print("Between images",0+1,1+1,"Before optimization Linear Triang: ", mean_err1, "After optimization Non-Linear Triang: ", mean_err2)
    
    # print(feature_x.shape)
    # print(feature_x.shape[0])
    "Resistering Cam 1 and 2"
    X_all = np.zeros((feature_x.shape[0],3))
    # print("new X all", X_all)
    cam_indices = np.zeros((feature_x.shape[0],1), dtype = int)
    X_found = np.zeros((feature_x.shape[0],1), dtype = int)
    # print("X_found_1", X_found)
    # print(idx)
    X_all[idx] = X[:,:3]
    X_found[idx] = 1
    # print("X_found_2", X_found)
    cam_indices[idx] = 1
    X_found[np.where(X_all[:2]<0)] = 0
    # print("X_found_3", X_found)

    C_set = []
    R_set = []

    C0 = np.zeros(3)
    R0 = np.identity(3)
    C_set.append(C0)
    R_set.append(R0)
    C_set.append(C_best)
    R_set.append(R_best)

    # feature_idx = np.where(X_found[:,0])
    # X = X_all[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]

    # feature_idx = np.where(X_found[:,0])
    # X = X_all[feature_idx]
    x_ = X_refined[:,0]
    y_ = X_refined[:,1]
    z_ = X_refined[:,2]

    fig = plt.figure(figsize = (30,30))
    plt.xlim(-10,10)
    plt.ylim(-5,15)
    plt.scatter(x,z,marker='.',linewidths=1.5, color = 'blue', label = 'linear')
    plt.scatter(x_,z_,marker='.',linewidths=1.5, color = 'red', label = 'nonlinear') # added this line to plot x_ and z_ data
    for i in range(0, len(C_set)):
        R1 = get_euler(R_set[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_set[i][0],C_set[i][2], marker=(3,0, int(R1[1])), markersize=15, linestyle='None')
    plt.legend(loc = 'upper left')
    plt.savefig(Output+'lin_non_lin.png')
    plt.show()

    print("#########Registered Cam 1 and Cam 2 ############")

    for i in range(2,5):
        print("Registering Image: ", str(i+1))
        feature_idx_i = np.where(X_found[:,0] & filtered_feature_flag[:,i])
        if len(feature_idx_i[0]) < 8:
            print("Got ", len(feature_idx_i), "common points between X and ", i, "image")
            continue

        pts_i = np.hstack((feature_x[feature_idx_i, i].reshape(-1,1), feature_y[feature_idx_i, i].reshape(-1,1)))
        X = X_all[feature_idx_i,:].reshape(-1,3)

        ##### Here We start PnP
        R_init, C_init = PnPRANSAC(K,pts_i,X, 1000, 5)
        linear_error_pnp = PnP_reprojection_error(X, pts_i, K, R_init, C_init)
        
        Ri, Ci = NonLinearPnP(K, pts_i, X, R_init, C_init)
        non_linear_error_pnp = PnP_reprojection_error(X, pts_i, K, Ri, Ci)
        print("Initial linear PnP error: ", linear_error_pnp, " Final Non-linear PnP error: ", non_linear_error_pnp)

        C_set.append(Ci)
        R_set.append(Ri)
        ###### WE start with the triangulation
        
        for k in range(0,i):
            idx_X_pts = np.where(filtered_feature_flag[:,k] & filtered_feature_flag[:,i])
            idx_X_pts = np.asarray(idx_X_pts)
            idx_X_pts = np.squeeze(idx_X_pts)

            if (len(idx_X_pts)<8):
                continue

            x1 = np.hstack((feature_x[idx_X_pts,k].reshape(-1,1), feature_y[idx_X_pts,k].reshape(-1,1)))
            x2 = np.hstack((feature_x[idx_X_pts,i].reshape(-1,1), feature_y[idx_X_pts,i].reshape(-1,1)))

            # print(x1.shape,x2.shape)
            # print(np.array(R_set[k]).shape,C_set[k])
            X_d = triangulate(K,R_set[k],C_set[k],Ri,Ci,x1,x2)
            # print("burr",X_d,X_d.shape)
            X_d = X_d/X_d[:,3].reshape(-1,1)
            # print("burr",X_d,X_d.shape)
            linear_err = []
            pts1 , pts2 = x1, x2
            for pt1, pt2, X_3d in zip(pts1,pts2,X_d):
                err1, err2 = reprojection_error(X_3d,pt1,pt2,R_set[k],C_set[k],Ri,Ci,K)
                linear_err.append(err1+err2)
    
            mean_linear_err = np.mean(linear_err)
            # print(mean_linear_err)
            
            X = non_linear_triangulation(K,x1,x2,X_d,R_set[k],C_set[k],Ri,Ci)
            # print(X.shape)
            X = X/X[:,3].reshape(-1,1)
            
            non_linear_err = []
            for pt1, pt2, X_3d in zip(pts1,pts2,X):
                err1, err2 = reprojection_error(X_3d,pt1,pt2,R_set[k],C_set[k],Ri,Ci,K)
                non_linear_err.append(err1+err2)
    
            mean_nonlinear_err = np.mean(non_linear_err)
            print("Linear Triang error: ", mean_linear_err, "Non-linear Triang error: ", mean_nonlinear_err)

            X_all[idx_X_pts] = X[:,:3]
            X_found[idx_X_pts] = 1

            print("Appended", idx_X_pts[0], "Points Between ", k, "and ",i )

            
            ##Visibility Matrix
            X_index, visibility_matrix = get_obs_index_and_viz_mat(X_found,filtered_feature_flag,nCam=i)
            # print(X_index,visibility_matrix)
            
            ##Bundle Adjustment
            print("########Bundle Adjustment Started")
            R_set_, C_set_, X_all_ = bundle_adjustment(X_index, visibility_matrix,X_all,X_found,feature_x,feature_y,filtered_feature_flag,R_set,C_set,K,nCam=i)
            # print(np.array(R_set).shape,np.array(C_set).shape,X_all.shape)
            
            for k in range(0,i+1):
                idx_X_pts = np.where(X_found[:,0] & filtered_feature_flag[:,k])
                x = np.hstack((feature_x[idx_X_pts,k].reshape(-1,1), feature_y[idx_X_pts,k].reshape(-1,1)))
                X = X_all_[idx_X_pts]
                BundAdj_error = PnP_reprojection_error(X,x,K,R_set_[k],C_set_[k])
                print("########Error after Bundle Adjustment: ", BundAdj_error)

            # print("############Registired camera: ", i+1,"############################")
    X_found_ = X_found
    X_found[X_all[:,2]<0] = 0
    X_found_[X_all_[:,2]<0] = 0
    print("#############DONE###################")

    feature_idx = np.where(X_found[:,0])
    X = X_all[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]

    feature_idx = np.where(X_found_[:,0])
    X_ = X_all_[feature_idx]
    x_ = X_[:,0]
    y_ = X_[:,1]
    z_ = X_[:,2]

    #####2D Plotting
    fig = plt.figure(figsize = (30,30))
    plt.xlim(-10,10)
    plt.ylim(-5,15)
    plt.scatter(x,z,marker='.',linewidths=1.5, color = 'blue', label = 'Before Sparse Bundle Adj')
    plt.scatter(x_,z_,marker='.',linewidths=1.5, color = 'red', label = 'After Sparse Bundle Adj') # added this line to plot x_ and z_ data
    for i in range(0, len(C_set_)):
        R1 = get_euler(R_set_[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_set_[i][0],C_set_[i][2], marker=(3,0, int(R1[1])), markersize=15, linestyle='None')
    plt.legend(loc = 'upper left')
    plt.savefig(Output+'2D.png')
    plt.show()

    # plt.xlim(-15,15)
    # plt.ylim(-5,25)
    # plt.scatter(x,z,marker='.',linewidths=0.5, color = 'blue')
    # for i in range(0, len(C_set_)):
    #     R1 = get_euler(R_set_[i])
    #     R1 = np.rad2deg(R1)
    #     plt.plot(C_set_[i][0],C_set_[i][2], marker=(3,0, int(R1[1])), markersize=15, linestyle='None')
    #     plt.plot(C_set[i][0], C_set[i][2], marker='o', markersize=5, color='red')

    # plt.savefig(Output+'2D.png')
    # plt.show()


    ######3D Plotting
    fig1= plt.figure(figsize= (5,5))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x_,y_,z_,color="green")

    plt.savefig(Output+'3D.png')
    plt.show()
    

if __name__ == '__main__':
    main()







