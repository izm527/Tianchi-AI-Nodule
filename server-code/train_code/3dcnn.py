# coding:utf-8
from keras.models import Model, Sequential, load_model
from keras.models import Sequential
from keras.layers import Input,Activation,Dense,Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D,UpSampling2D
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import MaxPool3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
import os
from glob import glob
import numpy as np


def my_metrics(y_true, y_pred):
	if y_true == 1 and y_pred == 1: 
            return 0.1
	else:
	    return 0


def cnn3d(inputs_shape):
    inputs = Input(inputs_shape, name='inputs')  # (bs,512,512)
    #conv1 = Convolution3D(36, 5,  activation='relu', border_mode='same')(inputs)
    #conv1 = Dropout(0.3)(conv1)
    #conv1 = Convolution3D(36, 3,  activation='relu', border_mode='same')(conv1)
    #pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    #conv2 = Convolution3D(48, 4,  activation='relu', border_mode='same')(pool1)
    ##conv2 = Dropout(0.3)(conv2)
    ##conv2 = Convolution3D(32, 3, 3, activation='relu', border_mode='same')(conv2)
    #pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    #           
    #conv3 = Convolution3D(64, 3,  activation='relu', border_mode='same')(pool2)
    ##conv3 = Dropout(0.3)(conv3)
    ##conv3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv3)
    #pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    #net = Flatten()(pool3)

    net = Flatten()(inputs)
    net = Dense(20,name='fc1')(net)
    #if bn:
    #    net = BatchNormalization(name='fc1-bn')(net)
    net = Activation('relu')(net)
    net = Dense(64,name='fc2')(net)
    net = Dense(1, name='fc4')(net)
    sigmoid = Activation('sigmoid')(net)

    model = Model(inputs=inputs, outputs=sigmoid)
    model.summary()
    return model
                         





def cnn3d1(inputs_shape, kernel_size=3, pool_size=2,bn=False):
    """
    y是one hot形式
    :param inputs_shape:
    :param kernel_size:
    :param pool_size:
    :return:
    """
    inputs = Input(inputs_shape, name='inputs')  # 49
    #net = BatchNormalization(name='bn1')(inputs)
    net = Conv3D(168, (3,3,3))(inputs)  # 49-5/1=44
    net = Dropout(0.8)(net)
    if bn:
        net = BatchNormalization(name='bn1')(net)
    net = Activation('relu')(net)
    net = MaxPool3D((2,2,2),2)(net)  # 22

    net = Conv3D(24, (3,3,3))(net)  # 20
    net = Dropout(0.7)(net)
    if bn:
        net = BatchNormalization(name='bn2')(net)
    net = Activation('relu')(net)
    net = MaxPool3D((2,2,2),2)(net)  # 10


    net = Conv3D(48, (3,3,3))(net)  # 8
    net = Dropout(0.6)(net)
    if bn:
     net = BatchNormalization(name='bn3')(net)
    net = Activation('relu')(net)
    net = MaxPool3D((2,2,2), 2)(net)  # 4

    net = Flatten()(net)
    net = Dense(100,name='fc1')(net)
    net = Dropout(0.7)(net)
    #if bn:
    #    net = BatchNormalization(name='fc1-bn')(net)
    net = Activation('relu')(net)
    net = Dense(64,name='fc2')(net)
    net = Dropout(0.5)(net)
    net = Activation('relu')(net)

    
    net = Dense(2, name='fc4')(net)
    net = Dropout(0.8)(net)
    #if bn:
    #    net = BatchNormalization(name='fc2-bn')(net)
    #net = Activation('relu',name='fc2-relu')(net)
    #net = Dense(1, name='fc3')(net)
    sigmoid = Activation('softmax')(net)
	
    model = Model(inputs=inputs, outputs=sigmoid)
    model.summary()
    return model


cnt = [0,0]
def read_data(prefix):
    data_list = glob(prefix + "delted/*")
    lb_list = glob(prefix + 'labels/*')
    data_list.sort()
    lb_list.sort()

    print 'files=', len(data_list)
#    print lb_list[:200]

    cnt = [0,0]
    all_data = []
    all_lb = []

    drop = 0
    for idx, nm in enumerate (data_list[:100]):
        ndls = np.load(nm)
        lbs = np.load(lb_list[idx])

        for jdx, ndl in enumerate (ndls):
	    if lbs[jdx] == 1:
            	all_data.append(ndl)
            	all_lb.append(1)
	    	cnt[1] += 1
                all_data.append(ndl.transpose([0,2,1]))
                all_data.append(ndl.transpose([2,1,0]))
                all_data.append(ndl.transpose([1,0,2]))
                all_data.append(ndl.transpose([1,2,0]))
                all_data.append(ndl.transpose([2,0,1]))
                all_lb.append(1)
                all_lb.append(1)
                all_lb.append(1)
                all_lb.append(1)
                all_lb.append(1)
	        cnt[1] += 5
	    else:
		drop += 1
		if drop % 4 != 3:
            	    all_data.append(ndl)
            	    all_lb.append(0)
	     	    cnt[0] += 1
		
		
    nn = len(all_data)    
    all_data = np.array([all_data]).transpose((1,2,3,4,0))
    all_data += 1000
    all_data /= 50
    all_data.dtype = int
    print np.max(all_data[2])
    print np.min(all_data[2])
    print np.max(all_data[8])
    print np.min(all_data[8])
    all_lb = to_categorical(np.array(all_lb), 2)
    all_lb = np.array(all_lb)
    print 'false/true: ', cnt
    print all_lb[0]
    return all_data[:nn*3/4], all_lb[:nn*3/4],all_data[nn*3/4:],all_lb[nn*3/4:]


model = cnn3d1([49, 49, 49, 1])
#model.load_weights('model/chp20')


prefix = 'classify/'
initial_lr = 0.0001
#initial_lr = 1e2
#optimizer = Adam(lr = initial_lr)
#optimizer = SGD(initial_lr) 
batch_size = 14
nb_epoch = 100 


train_data, train_lb, test_data, test_lb = read_data(prefix)
print train_data.shape, train_lb.shape, test_data.shape, test_lb.shape

#model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy']) #loss='binary_crossentropy')

for i in range(100):

    if i>=0:
        #opt = Adam(lr=initial_lr)
	opt = SGD(lr=initial_lr )#,  momentum=0.2, nesterov=True)
    	#model.compile(optimizer=opt, loss='binary_crossentropy') #loss='binary_crossentropy')
    	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) #loss='binary_crossentropy')
        initial_lr *= 0.95
    print 'eph-------------------', i, initial_lr
    model.fit(train_data, train_lb, batch_size = batch_size, epochs = 1, shuffle = True, 
	class_weight={0:0.1, 1:0.9}, validation_data = (test_data, test_lb))
    if i%10 == 0:
	model.save_weights('model/chp'+str(i))
	print 'save model-----------------'

