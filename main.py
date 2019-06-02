from keras.callbacks import ModelCheckpoint
from keras.models import Model

from PIL import Image

import h5py

from dp import trainGenerator, testGenerator, saveResult
from model import unet
from history import LossHistory

# config constant
train_path = '/data/projects/punim0619/yfdata/trainingset'
val_path = '/data/projects/punim0619/yfdata/testset'
test_path = '/data/projects/punim0619/yfdata/testset/test/'
test_image_type = 'tif'
image_folder = 'train'
val_image_folder = 'test'
mask_folder = 'mask'
image_color_mode = 'grayscale'
mask_color_mode = 'grayscale'
target_size = (256, 256)
batch_size = 8
save_to_dir = None
image_save_prefix  = 'after_train'
mask_save_prefix  = 'after_mask'
seed = 1
result_image_path = '/home/yfedward/capstone/unet-study/loss.jpg'
model_path = '/home/yfedward/capstone/unet-study/unet_best.hdf5'
result_save_path = '/data/projects/punim0619/yfdata/testset/result'

def main():
    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    img_gen_arg_dict = dict(directory = train_path,
                        classes = [image_folder],
                        class_mode = None,
                        color_mode = image_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        save_to_dir = save_to_dir,
                        save_prefix  = image_save_prefix,
                        seed = seed)

    mask_gen_arg_dict = dict(directory = train_path,
                        classes = [mask_folder],
                        class_mode = None,
                        color_mode = mask_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        save_to_dir = save_to_dir,
                        save_prefix  = mask_save_prefix,
                        seed = seed)
    val_img_gen_arg_dict = dict(directory = val_path,
                        classes = [val_image_folder],
                        class_mode = None,
                        color_mode = image_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        save_to_dir = save_to_dir,
                        save_prefix  = image_save_prefix,
                        seed = seed)
    val_mask_gen_arg_dict = dict(directory = val_path,
                        classes = [mask_folder],
                        class_mode = None,
                        color_mode = mask_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        save_to_dir = save_to_dir,
                        save_prefix  = mask_save_prefix,
                        seed = seed)    
    myGene = trainGenerator(data_gen_args, img_gen_arg_dict, mask_gen_arg_dict)
    valGene = trainGenerator(data_gen_args, val_img_gen_arg_dict, val_mask_gen_arg_dict)

    history = LossHistory()
    model = unet()
    model_checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=300, epochs=10, validation_data=valGene, validation_steps=30, callbacks=[model_checkpoint, history])

    history.loss_plot('epoch', result_image_path)
    print('result saved')

    testGene = testGenerator(test_path, test_image_type)
    results = model.predict_generator(testGene, 10, verbose=1)
    saveResult(result_save_path, results)

if __name__ == '__main__':
    main()
