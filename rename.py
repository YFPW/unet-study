import os

prefix = 'img'
start_img = 20
img_number = 2
crop_number = 16
test_prefix = 'test_img'

for i in range(img_number):
    for j in range(crop_number):
        origin_name = prefix + str(start_img + i) + '_c' + str(j) + '.tif'
        new_name = test_prefix + str(i * crop_number + j) + '.tif'
        os.rename(origin_name, new_name)
