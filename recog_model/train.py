import numpy as np
import talos as ta
import pickle
from keras.applications import InceptionResNetV2
from keras import layers
from keras import models
from keras import optimizers
from keras.utils.np_utils import to_categorical

history = None

def prepare_model(x_train,y_train,x_val,y_val,params):
    conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation=params['activation']))
    model.add(layers.Dense(5, activation='sigmoid'))
    conv_base.trainable = False
    model.compile(loss=params['losses'], optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
    validation_data = [x_val,y_val]
    history = model.fit(x_train,y_train,
                        steps_per_epoch=params['batch_size'],
                        epochs=params['epochs'],
                        validation_data=validation_data,
                        validation_steps=params['batch_size'])
    #self.history = history
    #self.model = model
    #self.conv_base = conv_base
    return history,model

class InceptionResnetV2ImageClassifier:

    hyper_params = {
    'activation':'relu',
    'losses': 'categorical_crossentropy',
    'batch_size': [],
    'epochs': 12
    }

    tr_imgs, tr_labels, val_imgs, val_labels =  [],[],[],[]

    def __init__(self,model_asset=""):
        assert len(model_asset) != 0 , "Invalid Image Classifier Name"
        self.m_asset = model_asset


    def define_data(self,tr_img_data=None,tr_labels=None,classes=0,val_img_data=None,val_labels=None):
        assert (tr_img_data.shape[0] != 0 and tr_labels.shape[0] != 0) , "Empty data not accepted"
        self.tr_imgs = tr_img_data
        self.tr_labels = to_categorical(tr_labels,num_classes=5)
        self.val_imgs = val_img_data
        self.val_labels = to_categorical(val_labels,num_classes=5)
        self.classes = classes
        self.hyper_params['batch_size'] = 8

    def prepare_model(self,x_train,y_train,x_val,y_val,params):
        conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation=params['activation']))
        model.add(layers.Dense(len(self.classes), activation='softmax'))
        conv_base.trainable = False
        model.compile(loss=params['losses'], optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])

        validation_data = [x_val,y_val]
        history = model.fit(x_train,y_train,
                            steps_per_epoch=params['batch_size'],
                            epochs=params['epochs'],
                            validation_data=validation_data,
                            validation_steps=params['batch_size'])
        #self.history = history
        #self.model = model
        return history,model

    def fit_model(self):
        print("BEGINNING....TALOS-KERAS Scan() ")
        print(self.tr_imgs.shape)
        print(self.tr_labels.shape)
        print(self.hyper_params)
        print(self.val_imgs.shape)
        print(self.val_labels.shape)
        
        #scan_object = ta.Scan(self.tr_imgs, self.tr_labels,model=prepare_model, params=self.hyper_params,x_val=self.val_imgs,y_val=self.val_labels)
        #print("DONE....TALOS-KERAS Scan() ")
        
        #self.model.save_weights('model/model_weights.h5')
        #self.model.save('model/model_keras.h5')
        #with open("model/"+self.m_asset+".pkl","wb") as mf:
            #pickle.dump(scan_object,mf)
        print("SAVED...MODELS")
        hist , model_d = prepare_model(self.tr_imgs,self.tr_labels,self.val_imgs,self.val_label,self.hyper_params)
        acc = hist.history['acc']
        val_acc = hist.history['val_acc']
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        print("Accuracy : "+acc+" Val Acc : "+val_acc+" Loss: "+loss+"val_los : "+val_loss)
        
        #p = ta.Evaluate(scan_object)
        #p.evaluate(self.val_imgs,self.val_labels,average='macro')
