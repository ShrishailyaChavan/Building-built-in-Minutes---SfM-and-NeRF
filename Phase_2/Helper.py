import numpy as np
import torch
import json
import os
from nerf_helper_utility.utils import *
from nerf_helper_utility.get_rays import *
from nerf_helper_utility.Renderer import *

def rgb_values(height, width, focal, pose, near_threshold, far_threshold, coarse_features, batch_size, N_encode, model, device):
    
    origin_rays, ray_origins, depth_val, query_points = get_rays(height, width, focal, pose, near_threshold, far_threshold, coarse_features, device)
    
    flat_query_pts = query_points.reshape((-1,3))
    
    encoded_points = encode_position(flat_query_pts, N_encode)
    
    batches = generate_mini_batch(encoded_points, batch_size = batch_size)
    
    predictions = []
    
    for batch in batches:
        predictions.append((model(batch)))

    radiance_field_flat = torch.cat(predictions, dim=0)
    unflat_shape = list(query_points.shape[:-1]) + [4]
    radiance = torch.reshape(radiance_field_flat, unflat_shape)
    logits_rgb, _, _ = render_function(radiance, origin_rays, depth_val)

    return logits_rgb            
  
def checkTestPath():
    path = 'NeRF_results'
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)

def read_resize_images(imagePaths):
    images = []
    for i in range(len(imagePaths)):
        image = plt.imread(imagePaths[i])
        image.resize((100,100,3))
        images.append(image)
    images = np.array(images)
    images = torch.from_numpy(images)
    return images

def get_image_data(jsonData, datasetPath):
	imagePaths = []
	
	c2ws = []
	for frame in jsonData["frames"]:
		imagePath = frame["file_path"]
		imagePath = imagePath.replace(".", datasetPath)
		imagePaths.append(f"{imagePath}.png")
		c2ws.append(frame["transform_matrix"])
	
	return imagePaths, c2ws

def load_json(jsonPath):
	with open(jsonPath, "r") as fp:
		data = json.load(fp)
	
	return data