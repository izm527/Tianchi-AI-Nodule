# coding:utf-8

from keras.models import Sequential
from keras.layers import Input,Activation,Dense
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import MaxPool3D
from keras.layers.normalization import BatchNormalization
import os
import numpy as np


def cnn3d(inputs_shape, kernel_size=3, pool_size=2,bn=True):
    """
    y是one hot形式
    :param inputs_shape:
    :param kernel_size:
    :param pool_size:
    :return:
    """
    inputs = Input(inputs_shape, name='inputs')  # 49
    net = Conv3D(32,5,1,'same',name='conv1')(inputs)  # 49-5/1=44
    if bn:
        net = BatchNormalization(name='bn1')(net)
    net = Activation('relu',name='relu1')(net)
    net = MaxPool3D((2,2,2),2)(net)  # 22

    net = Conv3D(64,3,1,'valid', name='conv2')(net)  # 20
    if bn:
        net = BatchNormalization(name='bn2')(net)
    net = Activation('relu',name='relu2')(net)
    net = MaxPool3D((2,2,2),2)(net)  # 10


    net = Conv3D(128,3,1,'valid', name='conv3')(net)  # 8
    if bn:
     net = BatchNormalization(name='bn3')(net)
    net = Activation('relu',name='relu3')(net)
    net = MaxPool3D((2,2,2),2)(net)  # 4

    net = Dense(1000,name='fc1')(net)
    if bn:
        net = BatchNormalization(name='fc1-bn')(net)
    net = Activation('relu',name='fc1-relu')(net)

    net = Dense(1000,name='fc2')(net)
    if bn:
        net = BatchNormalization(name='fc2-bn')(net)
    net = Activation('relu',name='fc2-relu')(net)
    net = Dense(2, name='fc3')(net)
    if bn:
        net = BatchNormalization(name='fc3-bn')(net)
    net = Activation('softmax', name='fc3-relu')(net)

    return net


def readdata(prefix):
    data_list = glob(prefix + "origin/*")
    lb_list = glob(prefix + 'labels/*')
    print data_list[:200]
    print lb_list[:200]

    all_data = []
    all_lb = []
    for idx, nm in enumerate (data_list):
        ndls = np.load(nm)
        lbs = np.load(lb_list[idx])
        for jdx, ndl in enumerate (ndls):
            all_data.append(ndl)
            all_lb.append(lbs[jdx])
    
    all_data = np.array(all_data)
    all_lb = np.array(all_lb)
    print all_data.shape, all_lb.shape



prefix = 'classify/'
model = cnn3d([None, 49, 49, 94])



initial_lr = 1e-5
optimizer = Adam(lr = initial_lr)
batch_size = 8
nb_epoch = 100

model.compile(optimizer=optimizer)
model.fit(train_data, train_lb, batch_size = batch_size, epochs = np_epoch, shuffle = True)

