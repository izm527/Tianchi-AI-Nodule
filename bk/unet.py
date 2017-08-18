# coding:utf-8
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.optimizers import SGD, Adam
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D,UpSampling2D
from keras.utils import np_utils
from keras import backend as K


def read_indexfile(indexfilepath):
    f = open(indexfilepath,'r')
    res = []
    for line in  f.readlines():
        line = line.strip('\n')
        line = line.replace('\\','/')
        res.append(line)
    return res


# change the loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet(inputs_shape,kernel_size=3,pool_size=2):
    """

    :param inputs_shape: tuple,e.g.(bs,512,512)
    :param kernel_size:
    :param pool_size:
    :return:
    """
    inputs = Input(inputs_shape,name='inputs')  # (bs,512,512)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.summary()
    return model


def solve(model,inputs,settings):

    optimizer = settings.get('optimizer', Adam(lr=1e-5))
    loss = settings.get('loss', dice_coef_loss)
    metrics = settings.get('metrics', [dice_coef])
    batch_size = settings.get('batch_size', 128)
    nb_epoch = settings.get('nb_epoch', 100)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    hist = model.fit(x=inputs[0], y=inputs[1], batch_size=batch_size, nb_epoch=nb_epoch, verbose=True,
                     validation_split=0.2, shuffle=True, validation_data=settings['valdata'])

def loadData(dataset_name):
    ###################################
    # 此处待改
    # 处理数据，原始数据可得到0,1值的数据，输出是[n,64,64,64,1]的数据，float32：0.0和1.0
    # data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
    # data = read_indexfile(os.path.join("./data", config.dataset+'.txt'))
    setname = dataset_name.split('-')
    if setname[0] == 'modelnet_binvox10':
        if os.path.isdir('/home/xuhaiyue/ssd/data'):
            # 服务器路径
            data = read_indexfile(
                os.path.join("/home/xuhaiyue/ssd/data/modelnet_binvox10", setname[1], 'obj_indexfile2.txt'))
        else:
            # 本地路径
            data = read_indexfile(os.path.join("/media/hy/source/workspace/data/modelnet_binvox10", setname[1],'obj_indexfile.txt'))
        ###################################
    elif setname[0] == '3dshapenet':
        if os.path.isdir('/home/xuhaiyue/ssd/data'):
            # 服务器路径
            data = read_indexfile(
                os.path.join("/home/xuhaiyue/ssd/data/3dshapenet", setname[1], 'obj_indexfile2.txt'))
        else:
            # 本地路径
            data = read_indexfile(os.path.join("/media/hy/source/workspace/data/3DShapeNets/volumetric_data", setname[1],'obj_indexfile.txt'))
        ###################################
    else:
        raise ValueError('No {}!'.format(dataset_name))
    return data



input_list = glob("output/*in*.mhd")
lb_list = glob("output/*lb*.mhd")


def main():
    print input_list
    print lb_list


'''
def main():
    dataset_name = ''

    optimizer = Adam(lr=1e-5)
    loss = dice_coef_loss
    metrics = [dice_coef]
    bs = 128
    nb_epochs = 100
    input_h,input_w = 512,512

    model = unet(inputs_shape=(bs,input_h,input_w))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print ("Loading {} data...".format(dataset_name))
    data = loadData(dataset_name)
    data = np.array(data)
    print ("Finished loading {} data...".format(dataset_name))

    for epoch in range(nb_epochs):
        # 读入bs数据
        batch_idxs = np.random.choice(len(data), bs, False)
        batch_objs = data[batch_idxs]
        batch = [get_obj(batch_obj) for batch_obj in batch_objs]
        batch_objs = np.array(batch).astype(np.float32)  # [n,64,64,64,1]
'''
