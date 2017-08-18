import os
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.optimizers import SGD, Adam
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D,UpSampling2D
from keras.layers import Conv2D
from keras.layers.merge import concatenate
from keras.utils import np_utils
from keras import backend as K
from glob import glob
import tensorflow as tf




#################
#	Defining some functions for train
#################
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
    smooth = 0.1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)




def settings():
    st = {}
    st['initial_lr'] = 1e-5
    st['optimizer'] = Adam(lr = st['initial_lr'])
    st['loss'] = dice_coef_loss
    st['metrics'] = dice_coef
    st['batch_size'] = 32
#    metrics = settings.get('metrics', [dice_coef])


#################
#	UNet-define
#################
def unet(inputs_shape, kernel_size=3, pool_size=2):
    """
    :param inputs_shape: tuple,e.g.(bs,512,512)
    :param kernel_size:
    :param pool_size:
    :return:
    """   
    inputs = Input(inputs_shape, name='inputs')  # (bs,512,512)
    conv1 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
               
    conv3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
                         

    conv4 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    
    conv5 = Convolution2D(48, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Convolution2D(48, 3, 3, activation='relu', border_mode='same')(conv5)

    
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Dropout(0.3)(conv8)
    conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Dropout(0.3)(conv9)
    conv9 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.summary()
    return model
     


def load_data(p, q):
    lb_prefix = "vessel/unet-lb-"
    input_prefix = "vessel/unet-in-"
    file_name = "label.txt"
    fp = open(file_name)
    lines = fp.readlines()
    valid_scan_name = []

    
    #only use the file marked '1'
    for line in lines:
        words = line.split()
        if words[1] == '0':
            valid_scan_name.append(words[0])
    
    print 'size=', len(valid_scan_name)
    #concat all the cts picture to form the big array
    data_mat = []
    lb_mat = []
    for pic in valid_scan_name[p:q]:
        ip = np.load(input_prefix+pic+'.npy')
	if ip.shape[1] != 512:
	    continue
	print ip.shape
        for slc in ip:
            data_mat.append([slc])
	            
        lb = np.load(lb_prefix+pic+'.npy')
	if lb.shape[1] != 512:
	    continue
        for slc in lb:
            lb_mat.append([slc])

    data_mat = np.array(data_mat)
    data_mat = data_mat.transpose([0,2,3,1])


    #binary the label
    lb_mat = np.array(lb_mat).transpose([0,2,3,1])
    lb_mat[lb_mat != -1000] = 1
    lb_mat[lb_mat == -1000] = 0

    return data_mat, lb_mat



################## main ops 
#totally there are 450 cases, [0-350] for train, [350-450] for test
data, lb = load_data(0,360)
shuffle_idx = range(data.shape[0])
data = data[shuffle_idx]
lb = lb[shuffle_idx]

test_data, test_lb = load_data(360,404)
shuffle_idx = range(test_data.shape[0])
test_data = test_data[shuffle_idx]
test_lb = test_lb[shuffle_idx]

print data.shape
print test_data.shape





def main():
    optimizer = Adam(lr=3e-4)
    loss = dice_coef_loss
    metrics = [dice_coef]
    bs = 14
    nb_epochs = 5
    input_h,input_w = 512,512

    model = unet(inputs_shape=(input_h,input_w, 1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)    
    model.load_weights('model/vessel/'+'0-370_batch-14_eph-5')

   
    print "start fitting"
    with tf.device("/cpu:0"):
        model.fit(data, lb, batch_size = bs, epochs = nb_epochs, validation_data=(test_data, test_lb))  
        save_name = "0-370_batch-"+str(bs)+"_eph-"+str(nb_epochs)       
        model.save_weights("model/vessel/"+save_name)

        print "model finish trianning and save weights: ", save_name

        
    return model

model = main()
