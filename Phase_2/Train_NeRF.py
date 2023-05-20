import torch
import argparse
from Train_Dependency import *
from nerf_helper_utility.utils import *

def main():
  """
        This method performs training lego data using NeRF network using the below arguments passed.

        Args:
        NumOfEpochs (int): Number of Epochs
        BatchSize (int): Number of training examples
        NearThreshold (int): Near threshold
        FarThreshold (int): Far threshold
  """

  # Release all the unused memory cached by the CUDA runtime
  torch.cuda.empty_cache()

  device = retrieve_device()

  # Parse command-line arguments
  Parser = argparse.ArgumentParser()
  Parser.add_argument('--NumOfEpochs', type=int, default=100000)
  Parser.add_argument('--CB', type=int, default=32)
  Parser.add_argument('--BatchSize', type=int, default=4096)
  Parser.add_argument('--NearThreshold', type=int, default=2)
  Parser.add_argument('--FarThreshold', type=int, default=6)

  Args = Parser.parse_args()

  # Load the tiny_nerf_data.npz data
  imageList, poseList, focalList = load_data(device)

  # select subset from tuple to retrieve height, width
  height, width = imageList.shape[1:3]

  # Train the NeRF with required params
  Training_NeRF(imageList, poseList, focalList, height, width, 5e-3, 6, Args.NumOfEpochs,
                Args.NearThreshold, Args.FarThreshold, Args.BatchSize, Args.CB, device)

# Set the device for computation

def retrieve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
  main()