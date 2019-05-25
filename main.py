from keras.callbacks import ModelCheckpoint
from keras.models import Model

from dp import trainGenerator

train_path = 'data/membrane/train'
image_folder = 'image'
mask_folder = 'label'
image_color_mode = 'grayscale'
mask_color_mode = 'grayscale'
target_size = (256,256)
batch_size = 2
save_to_dir = None
image_save_prefix  = 'image'
mask_save_prefix  = 'mask'
seed = 1

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

    mask_generator = dict(directory = train_path,
                        classes = [mask_folder],
                        class_mode = None,
                        color_mode = mask_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        save_to_dir = save_to_dir,
                        save_prefix  = mask_save_prefix,
                        seed = seed)
                                                
    myGene = trainGenerator(data_gen_args, img_gen_arg_dict, mask_gen_arg_dict)

    model = unet()
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

    #testGene = testGenerator("data/membrane/test")
    #results = model.predict_generator(testGene,30,verbose=1)
    #saveResult("data/membrane/test",results)
