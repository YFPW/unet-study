from skimage import io
import numpy as np

im = io.imread('test_img0.tif')
print(im.shape)
print(im.dtype)
print(im.size)
#io.imshow(im)
print(im)

maxPix = -1
minPix = -1
for row in im:
    for col in row:
        for pix in col:
            if maxPix == -1 and minPix == -1:
                maxPix = pix
                minPix = pix
            else:
                if pix > maxPix:
                    maxPix = pix
                if pix < minPix:
                    minPix = pix
print('Max pixel: %d' %(maxPix))
print('Min pixel: %d' %(minPix))

im = im / 8
im = im.astype(np.uint8)
print(im.dtype)
print(im)
#io.imshow(im)
io.imsave('test_img_mod0.tif', im)
