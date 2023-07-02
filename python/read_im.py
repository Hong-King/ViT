import os
#import cv2
from PIL import Image

def read_im(path, endpoint = None):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            folder_name = os.path.basename(os.path.dirname(file_path))
            img_name = os.path.basename(file_path)
            img = Image.open(file_path)
            print(folder_name, img_name, img.size)

path =  "D:\CV\Codes\ViT\data_test"
read_im(path)