from skimage import io
import numpy as np
import glob as gb
import sys
import getopt
import os

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
    print('------------------------------------------options------------------------------------------')
    print('image_type: %s' %(image_type))
    print('input_dir: %s' %(input_dir))
    print('output_dir: %s' %(output_dir))
    print('width: %d' %(width))
    print('height: %d' %(height))
    print('---------------------------------------options end-----------------------------------------')
    

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_input_files():
    file_path = os.path.join(input_dir, '*.' + image_type)
    return sorted(gb.glob(file_path))

def max_and_min(img):
    maxPix = -1
    minPix = -1
    for row in img:
        for pix in row:
            if maxPix == -1 and minPix == -1:
                maxPix = pix
                minPix = pix
            else:
                if pix > maxPix:
                    maxPix = pix
                if pix < minPix:
                    minPix = pix
    return maxPix, minPix


def imgsrgb2grey(img_path):
    print('Number of images: %d' %(len(img_path)))
    #maxPix = -1
    #minPix = -1
    for path in img_path:
        img = io.imread(path)
        img = rgb2gray(img)
        img = img / 10
        img = img.astype(np.uint8)
        print(path)
        (img_dir, file_name) = os.path.split(path)
        savepath = os.path.join(output_dir, file_name)
        io.imsave(savepath, img)
        print(savepath)
        #print(img.shape)
        #print(img.dtype)
        '''
        img_max, img_min = max_and_min(img)
        if maxPix == -1 and minPix == -1:
            maxPix = img_max
            minPix = img_min
        else:
            if img_max > maxPix:
                maxPix = img_max
            if img_min < minPix:
                minPix = img_min
    print('Max pixel: %d' %(maxPix))
    print('Min pixel: %d' %(minPix))
    '''

if __name__ == '__main__':
    get_self_opt(sys.argv[1:])
    input_files = get_input_files()
    mkdir(output_dir)
    imgsrgb2grey(input_files)


