import numpy as np
import os
from PIL import Image
import sys

# load data

def load_im(path, endpoint = None):
    i = 0
    images = np.zeros(shape=(512,512,3,30))
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # folder_name = os.path.basename(os.path.dirname(file_path))
            # img_name = os.path.basename(file_path)
            img = Image.open(file_path)
            img = img.resize((512,512))

            images[:,:,:,i] = np.array(img)
            i = i + 1

            # print(folder_name, img_name, img.size)
    return images
        

path =  "D:\CV\Codes\ViT\data_test"

images = load_im(path)

img = np.zeros(shape=(512,512,3))
img = images[:,:,:,29]



















im = Image.fromarray(img.astype(np.uint8),mode='RGB')
im.show()



            
