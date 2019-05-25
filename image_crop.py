import os
import sys
import getopt
import glob as gb
import numpy as np
from PIL import Image
from scipy import misc
from libtiff import TIFF
import tifffile as tiff

def get_self_opt(argv):
    
    global image_type
    global input_dir
    global output_dir
    global width
    global height
    global prefix

    input_type = 'tif'
    input_dir = '~'
    output_dir = '~/output'
    width = 512
    height = 512
    prefix = 'img'

    try:
        opts, args = getopt.getopt(argv, "ht:i:o:w:l:p:", ["help", "type=", "input=", "output=", "width=", "height=", "prefix="])
    except getopt.GetoptError:
        print('Options error!')
        print('Error: image_crop.py -t <image_type> -i <input_dir> -o <output_dir> -w <image_width> -h <image_height> -p <prefix>')
        print('   or: image_crop.py --type=<image_type> --input=<input_dir> --output=<output_dir> --width=<image_width> --height=<image_height> --prefix=<prefix>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('    image_crop.py -t <image_type> -i <input_dir> -o <output_dir> -w <image_width> -h <image_height> -p <prefix>')
            print('or: image_crop.py --type=<image_type> --input=<input_dir> --output=<output_dir> --width=<image_width> --height=<image_height> --prefix=<prefix>')
            sys.exit()
        elif opt in ("-t", "--type"):
            image_type = arg
        elif opt in ("-i", "--input"):
            input_dir = arg
        elif opt in ("-o", "--output"):
            output_dir = arg
        elif opt in ("-w", "--width"):
            width = int(arg)
        elif opt in ("-l", "--height"):
            height = int(arg)
        elif opt in ("-p", "--prefix"):
            prefix = arg

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def crop_image(path, count):
    # 分离文件目录，文件名及文件后缀
    (img_dir, tempfilename) = os.path.split(path)

    # Read image
    print('file name: ' + tempfilename)
    img = tiff.imread(path)
    #print(img)

    # Deal with images in seg
    if len(img.shape) > 3:
        img = img[0][0][0]
    tmp_img_height = img.shape[0]
    tmp_img_width = img.shape[1]
    print('Shape: ' + str(img.shape))

    # 对图像进行处理
    crop_count  = 0
    for h_start in range(0, tmp_img_height, height):
        for w_start in range(0, tmp_img_width, width):
            # Crop image
            h_crop = height if h_start + height <= tmp_img_height else tmp_img_height - h_start
            w_crop = width if w_start + width <= tmp_img_width else tmp_img_width - w_start
            cropped_image = img[h_start : h_start + h_crop, w_start : w_start + w_crop]

            # image padding
            bottom_padding = height - h_crop
            right_padding = width - w_crop
            BLACK = 0
            padded_image = None

            # 3d array
            if len(img.shape) == 3:
                padded_image = np.lib.pad(cropped_image, ((0, bottom_padding),(0, right_padding),(0, 0)), 'constant', constant_values=(BLACK))
            # 2d array
            else:
                padded_image = np.lib.pad(cropped_image, ((0, bottom_padding),(0, right_padding)), 'constant', constant_values=(BLACK))

            # Save cropped images
            cropped_file_name = prefix + str(count) + '_c' + str(crop_count) + '.' + image_type
            savepath = os.path.join(output_dir, cropped_file_name)
            tiff.imsave(savepath, padded_image)
            crop_count += 1
    print('%d images after cropping.' %(crop_count))
            

if __name__ == '__main__':

    get_self_opt(sys.argv[1:])
    
    # image_path stores all names of images
    image_path = os.path.join(input_dir, '*.' + image_type)
    print('------------------------------------------options------------------------------------------')
    print('image_type: %s' %(image_type))
    print('input_dir: %s' %(input_dir))
    print('output_dir: %s' %(output_dir))
    print('width: %d' %(width))
    print('height: %d' %(height))
    print('images: %s' %(image_path)) 
    print('---------------------------------------options end-----------------------------------------')
    
    # img_path stores all the images
    img_path = sorted(gb.glob(image_path))
    print('Number of images: %d' %(len(img_path)))

    mkdir(output_dir)

    count = 0
    for path in img_path:
        crop_image(path, count)
        count += 1
       
