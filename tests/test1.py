import os
import cv2
import numpy as np
#from imutils import face_utils
import pickle
from sklearn.preprocessing import LabelEncoder
from augment.image import ImageAugmentation
from augment.tweak import crop,resize,rotate_by_angle
import tensorflow as tf
#import mtcnn.mtcnn as MTCNN
from recog_model.train import InceptionResnetV2ImageClassifier
#from recog_model.inference import ImageInferenceEngine

train_dir = "dataset/train"
val_dir = "dataset/val"

modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"

labelencoder_y_1 = LabelEncoder()
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

usr_map =  {}

"""
def detect_faces(img):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]
    if confidence > 0.5:
        x1 = int(detections[0, 0, 0, 3] * w)
        y1 = int(detections[0, 0, 0, 4] * h)
        x2 = int(detections[0, 0, 0, 5] * w)
        y2 = int(detections[0, 0, 0, 6] * h)
        predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
        rect = dlib.rectangle(x1,y1,x2,y2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        return [{
            'box' : [x1,y1,x2,y2],
            'keypoints' : {
                'left_eye' : ( int((shape[0][0]+shape[1][0])/2) , int((shape[0][1]+shape[1][1])/2) ),
                'right_eye' : ( int((shape[2][0]+shape[3][0])/2) , int((shape[2][1]+shape[3][1])/2) )
                }
            }]

    return None

def generate_augmeted_images():
    o_mtcnn = MTCNN.MTCNN()
    tr_labels , tr_imgs = [] , []
    celebs = 0
    print("LOADING....TRAINING SET")
    for celeb in os.listdir(train_dir):
        celeb_train = os.path.join(train_dir,celeb)
        celebs = celebs + 1
        usr_map[celeb] = celebs
        for train_img in os.listdir(celeb_train):
            img_dir = os.path.join(celeb_train,train_img)
            #d = o_mtcnn.detect_faces(cv2.imread(img_dir))
            d = detect_faces(cv2.imread(img_dir))
            bbox = d[0]['box']
            keypoints = d[0]['keypoints']
            dY = keypoints['right_eye'][1] - keypoints['left_eye'][1]
            dX = keypoints['right_eye'][0] - keypoints['left_eye'][0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            img = cv2.imread(img_dir)
            img = rotate_by_angle(img,angle)
            width = bbox[2]
            height = bbox[3]
            f_img = resize(crop(img,[width,height],cords=(bbox[0],bbox[1])),(150,150))
            tr_imgs.append(f_img)
            tr_labels.append(celeb)
            augmentation = ImageAugmentation(f_img)
            augmented_imgs = augmentation.process()
            for imgs in augmented_imgs:
                for img in imgs:
                    tr_imgs.append(img[0])
                    tr_labels.append(celeb)
                    cv2.imwrite(img_dir[:-4]+"_"+img[1]+".jpg",img[0])
    print("LOADED....TRAINING SET")
    tr_labels = labelencoder_y_1.fit_transform(tr_labels)
    return (tr_imgs,tr_labels,celebs)


def load_validation_set():
    #o_mtcnn = MTCNN.MTCNN()
    val_imgs,val_labels = [] , []
    print("LOADING....VALIDATION SET")
    for celeb in os.listdir(val_dir):
        celeb_dir = os.path.join(val_dir,celeb)
        for val_img in os.listdir(celeb_dir):
            img_dir = os.path.join(celeb_dir,val_img)
            img = cv2.imread(img_dir)
            d = detect_faces(img)
            #d = o_mtcnn.detect_faces(img)
            bbox = d[0]['box']
            keypoints = d[0]['keypoints']
            dY = keypoints['right_eye'][1] - keypoints['left_eye'][1]
            dX = keypoints['right_eye'][0] - keypoints['left_eye'][0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            img = rotate_by_angle(img,angle)
            width = bbox[2]
            height = bbox[3]
            f_img = resize(crop(img,[width,height],cords=(bbox[0],bbox[1])),(150,150))
            val_imgs.append(f_img)
            val_labels.append(celeb)
    print("LOADED...VALIDATION SET")
    val_labels = labelencoder_y_1.fit_transform(val_labels)
    return (val_imgs,val_labels)
"""

def test():
    tf.logging.set_verbosity(tf.logging.ERROR)
    celebs = 5
    tr_imgs,tr_labels = None, None
    val_imgs,val_labels = None, None
    #print("LOADING....InceptionResnetV2ImageClassifier")
    print("LOADING DATASETS")
    with open("dataset/tr_imgs.pkl","rb") as f :
        tr_imgs = np.array(pickle.load(f))
    with open("dataset/tr_labels.pkl","rb") as f :
        tr_labels = np.array(pickle.load(f))
    with open("dataset/var_imgs.pkl","rb") as f :
        val_imgs = np.array(pickle.load(f))
    with open("dataset/var_labels.pkl","rb") as f :
        val_labels = np.array(pickle.load(f))
    print("LOADED DATASETS")
    print("LOADING....InceptionResnetV2ImageClassifier")
    irv2 = InceptionResnetV2ImageClassifier(model_asset="Face_Recognizer")
    print("LOADING....TRAINING AND VALIDATION SETS")
    irv2.define_data(tr_img_data=np.array(tr_imgs),tr_labels=np.array(tr_labels),classes=celebs,val_img_data=np.array(val_imgs),val_labels=np.array(val_labels))
    print("LOADING....FITTING IRV2 MODEL")
    irv2.fit_model()
