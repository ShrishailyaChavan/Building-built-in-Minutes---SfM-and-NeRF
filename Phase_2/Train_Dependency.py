import torch
import random
from Helper import *
from nerf_helper_utility.NeRF_Network import *

def Training_NeRF(imageList, poseList, focalList, height, width, lRate, noise_encode, numOfEpochs,\
                     near_threshold, far_threshold, batch_size, coarse_features, device):
  
  lossList = []
  epochList = []
  iteration = 50

  model = Nerf().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr = lRate)
  
  for i in range(numOfEpochs):
    image_index = random.randint(0, imageList.shape[0]-1)
    
    image = imageList[image_index].to(device)
    pose = poseList[image_index].to(device)

    rgb = rgb_values(height, width, focalList, pose, near_threshold, far_threshold, coarse_features, batch_size, noise_encode, model, device)
    photometricLoss = F.mse_loss(rgb, image) 
    photometricLoss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % iteration == 0:
      rgb = rgb_values(height, width, focalList, pose, near_threshold, far_threshold, coarse_features, batch_size, noise_encode, model, device)
      photometricLoss = F.mse_loss(rgb, image)
      print("Photometric Loss value :", photometricLoss.item())
      lossList.append(photometricLoss.item())
      epochList.append(i+1)
  plot_loss(epochList, lossList)
                
  torch.save({'epoch': epochList,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},'./model.ckpt')