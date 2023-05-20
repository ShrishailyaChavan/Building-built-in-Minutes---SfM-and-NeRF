import torch

def get_rays(height, width, focal, pose, near_threshold, far_threshold, coarse_features, device):
  #create a tensor of equally spaced values
  x_cord = torch.linspace(0, width-1, width)
  y_cord = torch.linspace(0, height-1, height)

  #create coordinate grids
  x_cartesian, y_cartesian = torch.meshgrid(x_cord, y_cord, indexing='xy')

  #shift the tensor mesh to CUDA or CPU device
  x_cartesian = x_cartesian.to(device)
  y_cartesian = y_cartesian.to(device)

  #convert pixel coordinate in an image to a normalized coordinate in 3D space
  normalise_x = (x_cartesian - width * 0.5) / focal
  normalise_y = (y_cartesian - height * 0.5) / focal

  #Stack a sequence of normalised tensors along a provided dimension to create a new tensor
  directions = torch.stack([normalise_x, - normalise_y, -torch.ones_like(x_cartesian)], dim = -1)
  directions = directions[..., None,:]

  #Rotate matrix
  rotation_matrix = pose[:3, :3]

  #Translate matrix
  translattion_matrix = pose[:3, -1]

  directions_camera = directions * rotation_matrix
  directions_ray = torch.sum(directions_camera, dim = -1)
  directions_ray = directions_ray/torch.linalg.norm(directions_ray, dim = -1, keepdims = True)

  #expand the translation_matrix shape to match directions_ray shape
  ray_origin =  torch.broadcast_to(translattion_matrix, directions_ray.shape)

  #create depth_value equally spaced (coarse_features points each) between range of near and far threshold 
  depth_value = torch.linspace(near_threshold, far_threshold, coarse_features)

  noise_shape_List = list(ray_origin.shape[:-1]) + [coarse_features]
  noise_compute = torch.rand(size = noise_shape_List) * (far_threshold - near_threshold)/coarse_features
  depth_value = depth_value + noise_compute
  depth_value = depth_value.to(device)
  direction_query_points = ray_origin[..., None, :] + directions_ray[..., None, :] * depth_value[..., :, None]

  return directions_ray, ray_origin, depth_value, direction_query_points