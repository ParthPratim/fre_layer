import numpy as np
import talos as ta
import pickle
from keras.applications import inception_resnet_v2
from keras import layers
from keras import models
from keras import optimizers


class InceptionResnetV2ImageClassifier:

    hyper_params = {
    'activation':['relu'],
    'losses': ['binary_crossentropy'],
    'batch_size': [],
    'epochs': [10,20]
    }

    tr_imgs, tr_labels, val_imgs, val_labels =  [],[],[],[],[]

    def __init__(self,model_asset=""):
        assert len(model_asset) == 0 , "Invalid Image Classifier Name"
        self.m_asset = model_asset


    def define_data(self,tr_img_data=None,tr_labels=None,classes=0,val_img_data=None,val_labels=None):
        assert (tr_img_data.shape[0] == 0 || tr_labels.shape[0] == 0) , "Empty data not accepted"
        self.tr_imgs = tr_img_data
        self.tr_labels = tr_labels
        self.val_imgs = val_img_data
        self.val_labels = val_labels
        self.classes = classes
        self.hyper_params['batch_size'] = len(self.labels)

    def prepare_model(self):
        conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation=self.hyper_params['activation']))
        model.add(layers.Dense(len(self.classes), activation='softmax'))
        conv_base.trainable = False
        model.compile(loss=self.hyper_params['losses'], optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
        validation_data = [self.val_imgs,self.val_labels]
        history = model.fit(self.tr_imgs,self.tr_labels,
                            steps_per_epoch=self.hyper_params['batch_size'],
                            epochs=self.hyper_params['epochs'],
                            validation_data=validation_data,
                            validation_steps=self.hyper_params['batch_size'])
        self.history = history
        self.model = model
        self.conv_base = conv_base
        return (history,model)

    def fit_model(self):
        scan_object = ta.Scan(x, y, model=prepare_model, params=p, grid_downsample=0.1)
        self.model.save_weights('model/model_wieghts.h5')
        self.model.save('model/model_keras.h5')
        with open("model/talos_keras.pkl","wb") as mf:
            pickle.dump(scan_object,mf)
        history = self.history
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        print("Accuracy : "+acc+" Val Acc : "+val_acc+" Loss: "+loss+"val_los : "+val_loss)
    
