from keras.callbacks import ModelCheckpoint
from keras.models import Model

from PIL import Image

import h5py

from dp import trainGenerator
from model import unet
from history import LossHistory

# config constant
train_path = '/data/projects/punim0619/yfdata/trainingset'
image_folder = 'train'
mask_folder = 'mask'
image_color_mode = 'grayscale'
mask_color_mode = 'grayscale'
target_size = (512, 512)
batch_size = 2
save_to_dir = None
image_save_prefix  = 'after_train'
mask_save_prefix  = 'after_mask'
seed = 1
result_image_path = '/home/yfedward/capstone/unet/loss.jpg'
model_path = '/home/yfedward/capstone/unet/unet_best.hdf5'

def main():
    data_gen_args = dict(rotation_range=2,
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
                                                
    myGene = trainGenerator(data_gen_args, img_gen_arg_dict, mask_gen_arg_dict)

    history = LossHistory()
    model = unet()
    model_checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint, history])

    history.loss_plot('epoch', result_image_path)

    #testGene = testGenerator("data/membrane/test")
    #results = model.predict_generator(testGene,30,verbose=1)
    #saveResult("data/membrane/test",results)

if __name__ == '__main__':
    main()
