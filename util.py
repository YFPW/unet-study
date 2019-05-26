import skimage.io as io
import glob as gb
import os
import re
import random

from PIL import Image, ImageEnhance

from skimage.morphology import disk
from skimage.filters.rank import enhance_contrast

ignore_number = [3, 7, 11, 12, 13, 14, 15]

def get_imgs_size(path, size, dtype):
    img = io.imread(path)
    #print(type(img))
    print(path)
    print(img.shape)
    print(img.dtype)
    if img.shape != size or img.dtype != dtype:
        print(img.shape)
        print(img.dtype)

def img_enhance_contrct(path, output_dir, input_grd_dir, output_grd_dir):
    (img_dir, tempfilename) = os.path.split(path)
    s_number = int(re.findall(r"\d+", tempfilename)[0])
    file_number = int(re.findall(r"\d+", tempfilename)[1])
    if file_number in ignore_number:
        return

    img = Image.open(path)
    img = ImageEnhance.Contrast(img).enhance(8)

    output_path = os.path.join(output_dir, tempfilename)
    img.save(output_path)
    output_grd_path = 'seg' + str(s_number) + '_c' + str(file_number) + '.tif'
    img = io.imread(os.path.join(input_grd_dir, output_grd_path))
    io.imsave(os.path.join(output_grd_dir, output_grd_path), img)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def test_format():
    input_dir = '../dataset/trainingset/mask'
    image_type = 'tif'
    image_path = os.path.join(input_dir, '*.' + image_type)
    img_path = sorted(gb.glob(image_path))
    size = (512, 512)
    dtype = 'uint8'

    for path in img_path:
        get_imgs_size(path, size, dtype)

def ec():
    input_dir = '../dataset/trainingset/img_crop_grey/'
    output_dir = '../dataset/trainingset/img_crop_grey_ec/'
    input_grd_dir = '../dataset/trainingset/groundtruth/'
    output_grd_dir = '../dataset/trainingset/groundtruth2/'
    image_type = 'tif'
    image_path = os.path.join(input_dir, '*.' + image_type)
    img_path = sorted(gb.glob(image_path))
    mkdir(output_dir)
    mkdir(output_grd_dir)
    for path in img_path:
        img_enhance_contrct(path, output_dir, input_grd_dir, output_grd_dir)

def get_training_set(ratio = 10):
    input_dir = '../dataset/trainingset/img_crop_grey_ec'
    input_grd_dir = '../dataset/trainingset/groundtruth2'
    output_train_dir = '../dataset/trainingset/train'
    output_mask_dir = '../dataset/trainingset/mask'
    output_test_dir = '../dataset/testset/test'
    output_test_mask_dir = '../dataset/testset/mask'
    mkdir(output_train_dir)
    mkdir(output_mask_dir)
    mkdir(output_test_dir)
    mkdir(output_test_mask_dir)
    input_path = os.path.join(input_dir, '*.tif')
    images = sorted(gb.glob(input_path))

    # Randomly pick test images
    test_num = int(len(images) / ratio) 
    train_num = len(images) - test_num

    random.seed(10)
    test_image_slice = random.sample(images, test_num)

    for path in images:
        (img_dir, tempfilename) = os.path.split(path)
        if path in test_image_slice:
            output_test_path = os.path.join(output_test_dir, tempfilename)
            file_index = re.findall(r"img(.*)", tempfilename)[0]
            mask_filename = 'seg' + file_index
            input_mask_path = os.path.join(input_grd_dir, mask_filename)
            output_test_mask_path = os.path.join(output_test_mask_dir, mask_filename)
            os.popen('cp ' + path + ' ' + output_test_path)
            os.popen('cp ' + input_mask_path + ' ' + output_test_mask_path)

        else:
            output_train_path = os.path.join(output_train_dir, tempfilename)
            file_index = re.findall(r"img(.*)", tempfilename)[0]
            mask_filename = 'seg' + file_index
            input_mask_path = os.path.join(input_grd_dir, mask_filename)
            output_mask_path = os.path.join(output_mask_dir, mask_filename)
            os.popen('cp ' + path + ' ' + output_train_path)
            os.popen('cp ' + input_mask_path + ' ' + output_mask_path)

if __name__ == '__main__':
    #test_format()
    #ec()
    get_training_set()
