import torch
from Helper import *
from nerf_helper_utility.NeRF_Network import *

def Testing_NeRF(images, poseList, focal, height, width, N_encode,\
                     near_threshold, far_threshold, batch_size, coarse_features, device):
    
    model = Nerf()
    lossList = []
    iterate = 10041
    checkPoint = torch.load('model.ckpt')
    model.load_state_dict(checkPoint['model_state_dict'])
    model.to(torch.float64)
    model = model.to(device)
   
    model.eval()
    
    for i in range(len(images)):
        image = images[i].to(device)
        pose = poseList[i].to(device)
        rgb = rgb_values(height, width, focal, pose, near_threshold, far_threshold, coarse_features, batch_size, N_encode, model, device)
        photometricLoss = F.mse_loss(rgb, image)
        lossList.append(photometricLoss)
        plt.imshow(rgb.detach().cpu().numpy())
        plt.savefig("NeRF_results/" + str(iterate) + ".png")
        iterate += 1