import torch
import torch.nn.functional as F

def render_function(radiance, origin_rays, depth_val):
  #Extract the channel from radiance field tensor and pass it to ReLU activation function
  sigma = F.relu(radiance[...,3])   

  #Apply sigmoid function to the radiance tensor
  x = torch.sigmoid(radiance[...,:3])

  expo = torch.tensor([1e10], dtype = origin_rays.dtype, device = origin_rays.device)

  concatenated_distribution = torch.cat((depth_val[...,1:] - depth_val[...,:-1], expo.expand(depth_val[...,:1].shape)), dim = -1)

  y = 1. - torch.exp(-sigma * concatenated_distribution)

  #transmittance
  transmittance = y * cumulative_product(1. - y + 1e-10)

  #RGB color result of n depth values provided
  rgb_Data = (transmittance[..., None] * x).sum(dim = -2)

  depth_Data = (transmittance * depth_val).sum(dim = -1)

  accumulation_Data = transmittance.sum(-1)

  return rgb_Data, depth_Data, accumulation_Data

#Compute Cumulative product of tensors
def cumulative_product(tensor):

  #Calculate the cumulative product of tensor and shift the tensor by 1 place towards left
  cum_product = torch.cumprod(tensor, dim=-1)
  cum_product = torch.roll(cum_product, 1, dims=-1)
  cum_product[..., 0] = 1.
  
  return cum_product