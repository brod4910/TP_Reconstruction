# Transparent Camera Image Reconstruction
Transparent imaging is the process of using lensless cameras combined with a transparent window to take incoherent images. However, using image reconstruction to create partially recognizable images is the task of this repo.

# Network Architecture
Using the Unet from [this paper](https://arxiv.org/abs/1505.04597), we'll be able to reconstruct images taken directly from a transparent camera. If this proof-of-concept holds after an intial training, then we'll be able to make slight modifications to the network to better suit our problem.

# Dataset
The dataset we are currently using is the [COCO dataset](http://cocodataset.org/#detection-2017). However, if this dataset proves to not work with our setup, then we are going to regress and start with the simple [MNIST dataset](http://yann.lecun.com/exdb/mnist/).