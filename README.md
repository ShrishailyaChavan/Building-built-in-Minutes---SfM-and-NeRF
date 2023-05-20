# RBE/CS549: P2 - Buildings built in minutes - SfM and NeRF

<<<<<<< HEAD
Implementing the SfM Pipeline

# cd <folder where codes are added>
-- open Terminal
-- run the following
> python3 Wrapper.py - <path\to\data\folder> - <path\to\save\output>

-- Outputs are saved in the folder Phase_1\Data\IntermediateOutputImages\
=======
## Phase 2 Deep Learning Approach

Implemented the original NERF method from [this](https://arxiv.org/abs/2003.08934) paper.
>>>>>>> d049754fcfc31666a9395dd87ba012435d755ef3

Pre-requisites : 

1. Python
2. CUDA
3. Dataset For NeRF -> Download the lego data for NeRF from the original author’s [link](https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a)


## Physical Interpretation of NERF

![Physical Interpretation of NERF](https://github.com/JaniC-WPI/RBE549_P2/blob/master/Phase_2/NeRF_Output/Model%20Structure.png)

## Synthetic Results

![Synthetic Results](https://github.com/JaniC-WPI/RBE549_P2/blob/master/Phase_2/NeRF_Output/Output_NeRF.gif)

## Photometric Loss

![Loss](https://github.com/JaniC-WPI/RBE549_P2/blob/master/Phase_2/NeRF_Output/Loss.png)

## Steps to run the code:

1. Clone the repository
   git clone https://github.com/JaniC-WPI/RBE549_P2.git
   
2. Dowload the Dataset for lego.  (Refer point 3 of Pre-requisites mentioned above)

## Training
1. Go to the directory named Phase 2

```sh
 cd Phase_2
```
     
2. Start training the NeRF model on GPU/CPU depending on the availability of the device

```sh
  python3 Train_NeRF.py
```
  

After implementing above two steps for training, checkpoint named model.ckpt will be created, then you can execute the below code for Testing the NeRF model to produce synthetic results.

## Testing
1. Keep the same directory as you kept for Training. 
2. Testing the NeRF model
 
```sh
    python3 Test_NeRF.py
```
  

As a result, an output video will be created and saved named as "Output_NeRF.mp4" and a loss graph will be saved in NeRF_Output folder.

## References

1. https://arxiv.org/pdf/2003.08934.pdf
2. https://github.com/facebookresearch/neuralvolumes
3. https://rbe549.github.io/spring2023/proj/p2/
