## build-in functions
import numpy as np
import os
from PIL import Image
import sys

## make wheels
from read_im import read_im

path =  "D:\CV\Codes\ViT\data_test"

images = read_im(path)

print(len(images))

# img = np.zeros(shape=(512,512,3))
img = images[29]



















im = Image.fromarray(img.astype(np.uint8),mode='RGB')
im.show()



            
