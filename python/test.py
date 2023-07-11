import numpy as np
import os
from PIL import Image
import torch

path = '1.JPEG'

img = Image.open(r'D:\CV\Codes\ViT\python\1.JPEG')
# img.show()

img = img.resize((512,512))
# img.show()

img1 = np.zeros(shape=(2,512,512,3))
img1[0,:,:,:] = np.array(img)


img2 = np.zeros(shape=(512,512,3))
img2 = img1[0,:,:,:]

# print(np.array(img))
print(img2.shape)

a = torch.tensor(img2)
a = a.transpose(0,2)
[c,v,b] = a.shape
print(c,v,b)
print(img2.shape)

im = Image.fromarray(img2.astype(np.uint8),mode='RGB')
im.show()

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


            
