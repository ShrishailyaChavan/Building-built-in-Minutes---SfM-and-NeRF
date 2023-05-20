import argparse
from Test_Dependency import *

def main():
    """
        This method performs testing of lego data. below command-line arguments are passed.

        Args:
        NumOfEpochs (int): Number of Epochs
        BatchSize (int): Number of training examples
        NearThreshold (int): Near threshold
        FarThreshold (int): Far threshold
  """

    # Set the device to CPU
    device = torch.device("cpu")

    # Create path to save the test results
    checkTestPath()

    # Load test data and resize images
    data = load_json('Data/lego/transforms_test.json')

    image_Paths, poseList = get_image_data(data, 'Data/lego')

    poseList = np.array(poseList)
    poseList = torch.from_numpy(poseList).to(device)
    image = read_resize_images(image_Paths)
    image = image.to(device)

    # Parse command-line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CB', type=int, default=32)
    Parser.add_argument('--BatchSize', type=int, default=500)
    Parser.add_argument('--NearThreshold', type=int, default=2)
    Parser.add_argument('--FarThreshold', type=int, default=6)

    Args = Parser.parse_args()

    # Create 1d array
    focal = np.array([138.8889])
    focal = torch.from_numpy(focal).to(device)

    height, width = image.shape[1:3]

    # Test the NeRF with required params
    Testing_NeRF(image[1:6], poseList[1:6], focal, height, width, 6,
                 Args.NearThreshold, Args.FarThreshold, Args.BatchSize, Args.CB, device)

    # Create resulting video
    create_video()

if __name__ == '__main__':
    main()