from keras.preprocessing.image import ImageDataGenerator
import glob as gb
import os
import numpy as np
import skimage.io as io
import skimage.transform as trans

def adjustData(img, mask):
    img = img / 255.0
    mask = mask / 255.0
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)

def trainGenerator(aug_dict, img_gen_arg_dict, mask_gen_arg_dict):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
        
        Usage: myGene = trainGenerator(data_gen_args, img_gen_arg_dict, mask_gen_arg_dict)
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
        
    image_generator = image_datagen.flow_from_directory(**img_gen_arg_dict)
    mask_generator = mask_datagen.flow_from_directory(**mask_gen_arg_dict)
        
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img, mask)
        yield (img,mask)

def testGenerator(test_path, test_image_type, target_size = (256,256)):
    path = os.path.join(test_path, '*.' + test_image_type)
    test_images = sorted(gb.glob(path))
    for test_image in test_images:
        img = io.imread(test_image, as_grey=True)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape + (1,))
        img = np.reshape(img,(1,) + img.shape)
        yield img
                                
def saveResult(save_path, npyfile, target_size = (512,512)):
    for i, img in enumerate(npyfile):
        img = img[0] if len(img.shape) == 4 else img
        img = img[:,:,0] if len(img.shape) == 3 else img
        io.imsave(os.path.join(save_path,"%d_predict.png" %(i)), img)
                

