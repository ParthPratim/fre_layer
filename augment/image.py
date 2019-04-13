import cv2
import numpy as np
from augment.tweak import *

class ImageAugmentation:
    def __init__(self,img):
        self.in_img =  img
        self.augmented = None

    def process(self):
        return (
        flip(self.in_img),
        rotate(self.in_img),
        scale(self.in_img),
        crop_aug(self.in_img),
        translate_aug(self.in_img),
        gamma_aug(self.in_img),
        (gaussian_aug(self.in_img),)
        )

"""ia = ImageAugmentation("../dataset/sample/Jose_Rosado.jpg").process()
x = 0
for imgs in ia:
    for img in imgs:
        print(img[1])
        cv2.imwrite("../dataset/sample/Jose_Rosado_"+img[1]+".jpg",img[0])
"""
