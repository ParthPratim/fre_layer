import numpy as np
import cv2
import random

def resize(in_img,size):
    v = in_img[:]
    size = (int(size[0]),int(size[1]))
    return cv2.resize(v,size, interpolation = cv2.INTER_AREA)
    # size = (x,y) i.e tupple

def flip(in_img):
    v = in_img[:]
    return (
    (np.flipud(v),"Flip_Up_Down"),
    (np.fliplr(v),"Flip_Left_Right"),
    )

def rotate_by_angle(in_img,angle):
    image_center = tuple(np.array(in_img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(in_img, rot_mat, in_img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return np.array(result)

def rotate(in_img):
    return (
    (rotate_by_angle(in_img,10),"Rotate_10_degree"),
    (rotate_by_angle(in_img,15),"Rotate_15_degree"),
    (rotate_by_angle(in_img,45),"Rotate_45_degree"),
    (rotate_by_angle(in_img,-45),"Rotate_inv45_degree"),
    (rotate_by_angle(in_img,-15),"Rotate_inv15_degree"),
    (rotate_by_angle(in_img,-10),"Rotate_inv10_degree"),
    (rotate_by_angle(in_img,90),"Rotate_90_degree"),
    (rotate_by_angle(in_img,-90),"Rotate_inv90_degree"),
    (rotate_by_angle(in_img,random.randint(20,44)),"Rotate_Rnd"),
    )

def crop(in_img,size,cords=(0,0)):
    v = in_img[:]
    return np.array(v[cords[1]:size[1],cords[0]:size[0]],dtype=np.uint8)

def scale(in_img):
    black_2 = np.zeros((45,105,3),dtype=np.uint8)
    black_2_2 = np.zeros((150,45,3),dtype=np.uint8)
    black_3 = np.zeros((30,120,3),dtype=np.uint8)
    black_3_2 = np.zeros((150,30,3),dtype=np.uint8)
    scale_1 = resize(in_img,[150*1.3,150*1.3])
    scale_2 = resize(in_img,[150*0.7,150*0.7])
    scale_3 = resize(in_img,[150*0.8,150*0.8])
    return (
    (crop(scale_1,(150,150)),"Scale_2.0"),
    (np.append(np.append(scale_2,black_2,axis=0),black_2_2,axis=1),"Scale_0.7"),
    (np.append(np.append(scale_3,black_3,axis=0),black_3_2,axis=1),"Scale_0.8"),
    )

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def gamma_aug(in_img):
    return (
    (adjust_gamma(in_img,gamma=0.3),"Gamma_0.3"),
    (adjust_gamma(in_img,gamma=0.5),"Gamma_0.5"),
    (adjust_gamma(in_img,gamma=2.0),"Gamma_2.0"),
    (adjust_gamma(in_img,gamma=3.0),"Gamma_3.0")
    )

def crop_aug(in_img):
    return (
    (resize(crop(in_img,(150,150),cords=(32,0)),[150,150]),"Resize_X:32"),
    (resize(crop(in_img,(150,150),cords=(0,32)),[150,150]),"Resize_Y:32"),
    (resize(crop(in_img,(118,150),cords=(0,0)),[150,150]),"Resize_X:218"),
    (resize(crop(in_img,(150,118),cords=(0,0)),[150,150]),"Resize_Y:218")
    )

def translate(in_img,M):
    (rows, cols) = in_img.shape[:2]
    res = cv2.warpAffine(in_img, M, (cols, rows))

    return np.array(res)

def translate_aug(in_img):
    return (
    (translate(in_img,np.float32([[1,0,30],[0,1,0]])),"Trans_1"),
    (translate(in_img,np.float32([[1,0,-30],[0,1,0]])),"Trans_2"),
    (translate(in_img,np.float32([[1,0,0],[0,1,30]])),"Trans_3"),
    (translate(in_img,np.float32([[1,0,0],[0,1,-30]])),"Trans_4")
    )

def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise

    return noisy_image

def gaussian_aug(in_img):

    return (add_gaussian_noise(in_img,35),"gaussian_noice")
