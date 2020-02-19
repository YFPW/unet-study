import skimage.io as io
import skimage.transform as trans
import glob as gb
import os
import re
import random
import numpy as np
import cv2
import time

from PIL import Image, ImageEnhance

from skimage.morphology import disk
from skimage.filters.rank import enhance_contrast

ignore_number = [3, 7, 11, 12, 13, 14, 15]

def get_imgs_size(path, size, dtype):
    img = io.imread(path)
    #print(path)
    #print(type(img))
    #print(img.shape)
    #print(img.dtype)
    #print(img.max())
    #print(img.min())
    #print(len(img[img == img.max()]))
    #print(len(img[img == img.min()]))
    #print(len(img[img == img.max()]) + len(img[img == img.min()]) == 512*512)
    if img.shape != size or img.dtype != dtype:
        print(path)
        print(img.shape)
        print(img.dtype)

def img_enhance_contrct(path, output_dir, input_grd_dir = None, output_grd_dir = None):
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

def test_img_enhance_contrct(path, output_dir, input_grd_dir = None, output_grd_dir = None):
    io.imsave(os.path.join(output_grd_dir, output_grd_path), img)
#size = comm.size
    (img_dir, tempfilename) = os.path.split(path)

    img = Image.open(path)
    img = ImageEnhance.Contrast(img).enhance(8)

    output_path = os.path.join(output_dir, tempfilename)
    img.save(output_path)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def test_format():
    input_dir = '/mnt/hgfs/vmware_share_dir/unet-dataset/dataset/trainingset/img_crop_grey'
    image_type = 'tif'
    image_path = os.path.join(input_dir, '*.' + image_type)
    img_path = sorted(gb.glob(image_path))
    size = (512, 512)
    dtype = 'uint8'
    
    for path in img_path:
        get_imgs_size(path, size, dtype)
    

def resize():
    input_dir = '../dataset/testset/bad_test'
    output_dir = '../dataset/testset/bad_test_2'
    input_image_type = '.png'
    output_image_type = '.tif'
    output_size = (512, 512)
    input_path = os.path.join(input_dir, '*' + input_image_type)
    paths = sorted(gb.glob(input_path))
    mkdir(output_dir)

    for path in paths:
        (img_dir, tempfilename) = os.path.split(path)
        img = io.imread(path)
        img = trans.resize(img, output_size)
        img = np.array(img)
        img = img * 255
        img = img.astype(np.uint8)
        re_str = r"(.*)" + input_image_type
        filename = re.findall(re_str, tempfilename)[0] + output_image_type
        output_path = os.path.join(output_dir, filename)
        io.imsave(output_path, img)


def ec():
    input_dir = '../dataset/testset/bad_test_2'
    output_dir = '../dataset/testset/test2/'
    input_grd_dir = None
    output_grd_dir = None
    image_type = 'tif'
    image_path = os.path.join(input_dir, '*.' + image_type)
    img_path = sorted(gb.glob(image_path))
    mkdir(output_dir)
    #mkdir(output_grd_dir)
    for path in img_path:
        #img_enhance_contrct(path, output_dir, input_grd_dir, output_grd_dir)
        test_img_enhance_contrct(path, output_dir)

def get_training_set(train_num = 30, test_num = 10):
    input_dir = '../dataset/origin/img_crop_grey_ec'
    input_grd_dir = '../dataset/origin/groundtruth2'
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
    random.seed(10)
    images = random.sample(images, train_num + test_num)
    test_image_slice = images[:10]

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

def mix_pictures(img_path1, img_path2, weight1, weight2, output_path):
    img1 = cv2.imread(img_path1, 0)
    img2 = cv2.imread(img_path2, 0)
    img_mix = cv2.addWeighted(img1, weight1, img2, weight2, 0)
    cv2.imwrite(output_path, img_mix)

def mix():
    input_bottom_dir = '../dataset/testset/test'
    input_top_dir = '../dataset/testset/badtest-sth-good/test2'
    bottom_weight = 0.2
    output_test_dir = '../dataset/testset/mix' + str(bottom_weight)
    mkdir(output_test_dir)

    bottom_imgs = sorted(gb.glob(os.path.join(input_bottom_dir, '*tif')))
    top_imgs = sorted(gb.glob(os.path.join(input_top_dir, '*tif')))

    for bottom_img in bottom_imgs:
        (img_dir, bottomfilename) = os.path.split(bottom_img)
        top_img = random.sample(top_imgs, 1)[0]
        (img_dir, topfilename) = os.path.split(top_img)
        mix_number = re.findall(r"\d+", topfilename)[0]
        mixfilename = re.findall(r"(.*).tif", bottomfilename)[0] + '_m' + str(mix_number) + '.tif'
        mix_pictures(bottom_img, top_img, bottom_weight, 1 - bottom_weight, os.path.join(output_test_dir, mixfilename))
        print(bottom_img)
        print(top_img)
        print(mixfilename)
        print('')

if __name__ == '__main__':
    #mix()
    test_format()
    #resize()
    #ec()
    #get_training_set()
