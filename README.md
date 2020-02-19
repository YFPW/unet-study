# unet-study
This project is based on u-net neural network to segment cells on photos. Then through the trained model, I tried to segment the cells on other types of photos which are not clear enough to be trained.

## 1. Files
The files can be divided into two parts, preparing images for u-net and the u-net training.

### 1.1 Preprocessing
*image_crop.py*: Use tiffile to crop images
*rename.py*: Rename images
*rgb2grey.py*: Convert rgb images into grey
*tiff_format.py*: Transform the images into tiff

### 1.2 Training and testing
*dp.py*: Set up the generators
*history.py*: Record the accuracy and loss during training
*main.py*: The entrance of this program
*model.py*: The u-net model
*util.py*: Try to mix two types of images to test the network

### 1.3 Essay
*Capstone_essay.pdf*

## 2. Usage
To use this network, few steps need to be take. 
1. Your images used for taining must be *.tif 
2. GPU is need 
3. Images used for tesing must consist with training data, which is configured in *main.py*.
