import torch
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import shutil

#Plot the Loss Graphs
def plot_loss(epoch, log_loss_data):
    plt.figure(figsize=(10, 4))
    plt.plot(epoch, log_loss_data)
    plt.title("Loss")
    plt.savefig("NeRF_Output/Loss.png")

#Generate Mini Batch
def generate_mini_batch(encoded_data, batch_size):
  return [encoded_data[i:i + batch_size] for i in range(0, encoded_data.shape[0], batch_size)]

#Create output in video format
def create_video():
  result_path = 'NeRF_results'
  clip = ImageSequenceClip(result_path, 3)
  clip.write_videofile('Output_NeRF.mp4')
  shutil.rmtree(result_path)

def encode_position(flat_query_points, encode_collection):
  result = [flat_query_points]
  for i in range(encode_collection):
    result.append(torch.sin((2.0**i) * flat_query_points))
    result.append(torch.cos((2.0**i) * flat_query_points))
  
  result = torch.cat(result, axis = -1)

  return result

#Load data from tiny_nerf and extract features
def load_data(device):
    dataFilePath = 'Data/tiny_nerf_data.npz'
    loaded_Data = np.load(dataFilePath)

    images = extract_data(loaded_Data, "images", device)
    poses = extract_data(loaded_Data, "poses", device)
    focal = extract_data(loaded_Data, "focal", device)

    return images, poses, focal

def extract_data(loaded_Data, keyName, device):
   keyName = loaded_Data[keyName]
   extracted_data = torch.from_numpy(keyName).to(device)
   return extracted_data