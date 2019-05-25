from keras.preprocessing.image import ImageDataGenerator

def adjustData(img, mask):
    img = img / 255
    mask = mask /255
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
                

