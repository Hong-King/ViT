import numpy as np
import os
from PIL import Image

def read_im(path, endpoint = None):
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            img = Image.open(file_path)
            img = img.resize((512,512))

            images.append(np.array(img))

    return images


# def read_im(path, endpoint = None):
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             folder_name = os.path.basename(os.path.dirname(file_path))
#             img_name = os.path.basename(file_path)
#             img = Image.open(file_path)
#             print(folder_name, img_name, img.size)

## TEST CODES
path =  "D:\CV\Codes\ViT\data_test"
a = read_im(path)
a = np.array(a)
print(a.shape)