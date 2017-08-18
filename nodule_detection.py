import os
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.optimizers import SGD, Adam
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D,UpSampling2D
from keras.layers import Conv2D
from keras.layers.merge import concatenate

from keras import backend as K
from glob import glob

def detection(inputs_shape, kernel_size=3, pool_size=2):
    inputs = Input(inputs_shape, name='inputs')  # (bs,512,512)

    conv1 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(inputs)
    dpr1 = Dropout(0.3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(dpr1)
    
    conv2 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(pool1)
    dpr2 = Dropout(0.3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(dpr2)
    
    conv3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool2)
    dpr3 = Dropout(0.3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(dpr3)

    conv4 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool3)
    dpr4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(dpr4)
    
    conv5 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool4)
    dpr5 = Dropout(0.3)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(dpr5)
    
    conv6 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool5)
    dpr6 = Dropout(0.3)(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(dpr6)
    
    conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool6)
    dpr7 = Dropout(0.3)(conv7)
    pool7 = MaxPooling2D(pool_size=(2, 2))(dpr7)
    
    flt = Flatten()(pool7)
    ds1 = Dense(128)(flt)
    ds2 = Dense(64)(ds1)
    ds3 = Dense(3)(ds2)
    
    model = Model(inputs=inputs, outputs=ds3)
    model.summary()
    optimizer = Adam(lr=2e-3)
    model.compile(optimizer=optimizer, loss=square_loss)

    return model


def square_loss(y_true, y_pred):
    return np.sum((y_true - y_pred)*(y_true - y_pred))/3
    

def load_and_train(dect_model):
    lb_file = open(label_name)
    lines = lb_file.readlines()
    for start in range(0, len(lines), load_batch):
        data = []
        lb = []
        for idx in range(start, min(start+load_batch, len(lines))):
            objs = lines[idx].split(' ')
            print objs[0]
            array = np.load(input_path+'nodule-slcs/zclip-'+objs[0]+'.npy').transpose(1,2,0) 
            data.append(array)
            lb.append([float(objs[3]), float(objs[2]), float(objs[4])-float(objs[1])*128])   #z-value subtract the idex*128
        data = np.array(data)
        lb = np.array(lb)

        dect_model.fit(x=data, y=lb, batch_size=batch_size, nb_epoch=1, shuffle=True)
    dect_model.save_weights("model/"+save_name)
    print "model finish trianing and saved: ", save_name
    return dect_model


model = detection([512,512,128])

input_path = '/media/izm/Normal/FJJ/'
load_batch = 12
batch_size = 4
label_name = input_path+'nodule-slcs/'+'nodule-area.txt'
save_name = 'detection'

model.load_weights('model/detection')
for i in range(1):
    model = load_and_train(model)
    print "Now ------------------------finish iteration ", i
             
            
