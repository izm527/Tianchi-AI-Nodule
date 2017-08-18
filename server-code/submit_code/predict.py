# -- coding: utf-8 --
#1.load libraries
#from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
import scipy.ndimage
from tqdm import tqdm #progress tube
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage import measure, morphology
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage as ndi

from keras.models import Model, Sequential, load_model
from keras.layers.convolutional import Conv3D
from keras.layers import Input, Dense, Dropout,Flatten, merge
from keras.optimizers import SGD, Adam
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D,UpSampling2D
from keras.layers import Conv2D,Activation, MaxPool3D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_segmented_lungs(im, show=False):  
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if show == True:
        plot = plt.subplot()
        plot.imshow(im, cmap=plt.cm.bone)
        plt.show()
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -230
    if show == True:
        plot = plt.subplot()
        plot.imshow(binary, cmap=plt.cm.bone) 
        plt.show()
    cleared = clear_border(binary)

    if show == True:
        plot = plt.subplot()
        plot.imshow(cleared, cmap=plt.cm.bone) 
        print "3"
        plt.show()

    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if show == True:
        plot = plt.subplot()
        plot.imshow(label_image, cmap=plt.cm.bone)
        plt.show()
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0

    binary = label_image > 0
    if show == True:
        plot = plt.subplot()
        plot.imshow(binary, cmap=plt.cm.bone)
        plt.show()
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if show == True:
        plot = plt.subplot()
        plot.imshow(binary, cmap=plt.cm.bone)
        plt.show()
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if show == True:
        plot = plt.subplot()
        plot.imshow(binary, cmap=plt.cm.bone)
        plt.show()
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if show == True:
        plot = plt.subplot()
        plot.imshow(binary, cmap=plt.cm.bone)
        plt.show()
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = -1000
    
    
    if show == True:
        plot = plt.subplot()
        plot.imshow(im, cmap=plt.cm.bone)
        plt.show()
    
    im[im < threshold] = -1000
    if show == True:
        plot = plt.subplot()
        plot.imshow(im, cmap=plt.cm.bone)
        plt.show()
    
    return im



#####
# remove the two biggest vessels in the lung
#####


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)


def filt_vessels(segmented_ct_scan):
    print "filting vessels!"
    selem = ball(2)
    
    segmented_ct_scan0 = segmented_ct_scan.copy() 
    linking_scan = segmented_ct_scan0.copy()
    linking_scan[linking_scan >= -350] = 1
    linking_scan[linking_scan < -350] = 0

    binary = binary_closing(linking_scan, selem)
    label_scan = label(binary, neighbors=8)
    #label_scan = label(segmented_ct_scan, neighbors=4) 

    rr =  regionprops(label_scan)
    areas = [r.area for r in rr]
    areas.sort()
    print "areas: ", areas[-6:-1]


    for r in rr:
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000

        rds = r.coords

        for c in rds:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)

            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
        if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
            for c in rds:
                segmented_ct_scan0[c[0], c[1], c[2]] = -1000
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))
    return segmented_ct_scan0


def plot_3d(image, name, threshold=-1000):
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]    
    #print len(measure.marching_cubes(p, threshold))
    verts, faces, a, b = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    #plt.savefig(outpath + 'pics/' + name +'.png')
    print "plotting : ", name
    #plt.show()
    plt.close()


def unet(inputs_shape, kernel_size=3, pool_size=2):
    """
    :param inputs_shape: tuple,e.g.(bs,512,512)
    :param kernel_size:
    :param pool_size:
    :return:
    """   
    inputs = Input(inputs_shape, name='inputs')  # (bs,512,512)
    conv1 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
               
    conv3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
                         

    conv4 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    
    conv5 = Convolution2D(48, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Convolution2D(48, 3, 3, activation='relu', border_mode='same')(conv5)

    
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Dropout(0.5)(conv8)
    conv8 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Dropout(0.5)(conv9)
    conv9 = Convolution2D(24, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.summary()
    return model
  

    
def seperate_nodule(origin, filt, unet_mask):
    selem = ball(2)
    segmented_ct_scan = unet_mask.copy()
    #segmented_ct_scan = segmented_ct_scan0.copy()
    segmented_ct_scan[segmented_ct_scan >= 0.9999] = 1
    segmented_ct_scan[segmented_ct_scan < 0.9999] = 0

    #binary = binary_closing(segmented_ct_scan, selem)
    binary = segmented_ct_scan
    label_scan = label(binary, neighbors=8)

    rr =  regionprops(label_scan)
    areas = [r.area for r in rr]
    areas.sort()
    
    
    areas = []
    orig_data = []
    filt_data = []
    res_lb = []
    res_center = []
    #traverse all the center of seperated condinate nodules
    print unet_mask.shape
    for r in rr:
        if r.area < 24 or r.area > 34000:    #the condinate nodule is too small or too big and skip it
            continue
            
        rds = r.coords
        
        #get the geometrical center of nthe nodule
        x = y = z = cnt = 0
        for point in rds:
            x += point[2]
            y += point[1]
            z += point[0]
            cnt += 1
        x = int(np.rint(x/cnt))
        y = int(np.rint(y/cnt))
        z = int(np.rint(z/cnt))
        
        #build a cube to get the seperated nodule
        cube = np.zeros([49,49,49], dtype = int)
        cube1 = np.zeros([49,49,49], dtype = int)
        cube[:][:][:] = -1000
        cube1[:][:][:] = -1000
        cnt = 0
        for point in rds:
            xx = int(np.rint((point[2]-x)+24))
            yy = int(np.rint((point[1]-y)+24))
            zz = int(np.rint((point[0]-z)+24))
            if xx>=0 and xx<49 and yy>=0 and yy<49 and zz>=0 and zz<49:
                cnt += 1
                cube[zz][yy][xx] = origin[z][y][x]
                cube1[zz][yy][xx] = filt[z][y][x]
        if cnt < 20 or np.max(cube) < -990:
            print 'noting! = ',rds
            continue
            
        orig_data.append(cube)
        filt_data.append(cube1)
        res_center.append([x, y, z])
        
        #judge whether the condinate nodule is the real nodule
    print 'condinate nodule:', len(orig_data)
    return np.array(orig_data), np.array(filt_data), np.array(res_center)



def cnn3d1(inputs_shape, kernel_size=3, pool_size=2,bn=False):
    """
    y是one hot形式
    :param inputs_shape:
    :param kernel_size:
    :param pool_size:
    :return:
    """
    inputs = Input(inputs_shape, name='inputs')  # 49
    net = BatchNormalization(name='bn1')(inputs)
    net = Conv3D(160, (3,3,3))(inputs)  # 49-5/1=44
    #net = Dropout(0.8)(net)
    if bn:
        net = BatchNormalization(name='bn1')(net)
    net = Activation('relu')(net)
    net = MaxPool3D((2,2,2),2)(net)  # 22

    net = Conv3D(16, (5,5,5))(net)  # 20
    net = Dropout(0.8)(net)
    if bn:
        net = BatchNormalization(name='bn2')(net)
    net = Activation('relu')(net)
    net = MaxPool3D((2,2,2),2)(net)  # 10


    net = Conv3D(64, (3,3,3))(net)  # 8
    net = Dropout(0.8)(net)
    if bn:
     net = BatchNormalization(name='bn3')(net)
    net = Activation('relu')(net)
    net = MaxPool3D((2,2,2), 2)(net)  # 4

    net = Flatten()(net)
    net = Dense(200,name='fc1')(net)
    net = Dropout(0.6)(net)
    #if bn:
    #    net = BatchNormalization(name='fc1-bn')(net)
    net = Activation('relu')(net)
    net = Dense(200,name='fc2')(net)
    net = Dropout(0.5)(net)
    net = Activation('relu')(net)

    
    net = Dense(1, name='fc4')(net)
    net = Dropout(0.5)(net)
    #if bn:
    #    net = BatchNormalization(name='fc2-bn')(net)
    #net = Activation('relu',name='fc2-relu')(net)
    #net = Dense(1, name='fc3')(net)

    sigmoid = Activation('sigmoid')(net)
	
    model = Model(inputs=inputs, outputs=sigmoid)
    model.summary()
    return model


def cnn3d(inputs_shape, kernel_size=3, pool_size=2,bn=False):
    """
    y是one hot形式
    :param inputs_shape:
    :param kernel_size:
    :param pool_size:
    :return:
    """
    inputs = Input(inputs_shape, name='inputs')  # 49
    net = BatchNormalization(name='bn1')(inputs)
    net = Conv3D(64, (3,3,3))(inputs)  # 49-5/1=44
    #net = Dropout(0.8)(net)
    if bn:
        net = BatchNormalization(name='bn1')(net)
    net = Activation('relu')(net)
    net = MaxPool3D((2,2,2),2)(net)  # 22

    net = Conv3D(48, (5,5,5))(net)  # 20
    net = Dropout(0.8)(net)
    if bn:
        net = BatchNormalization(name='bn2')(net)
    net = Activation('relu')(net)
    net = MaxPool3D((2,2,2),2)(net)  # 10


    #net = Conv3D(32, (3,3,3))(net)  # 8
    ##net = Dropout(0.8)(net)
    #if bn:
    # net = BatchNormalization(name='bn3')(net)
    #net = Activation('relu')(net)
    #net = MaxPool3D((2,2,2), 2)(net)  # 4

    net = Flatten()(net)
    net = Dense(400,name='fc1')(net)
    net = Dropout(0.6)(net)
    #if bn:
    #    net = BatchNormalization(name='fc1-bn')(net)
    net = Activation('relu')(net)
    net = Dense(200,name='fc2')(net)
    net = Dropout(0.5)(net)
    net = Activation('relu')(net)

    
    net = Dense(1, name='fc4')(net)
    net = Dropout(0.5)(net)
    #if bn:
    #    net = BatchNormalization(name='fc2-bn')(net)
    #net = Activation('relu',name='fc2-relu')(net)
    #net = Dense(1, name='fc3')(net)

    sigmoid = Activation('sigmoid')(net)
	
    model = Model(inputs=inputs, outputs=sigmoid)
    model.summary()
    return model



def do_predict(model, data):
	data = data.reshape((data.shape[0], 49, 49, 49, 1))		
	res = model.predict(data, batch_size=4)
	return res




'''
model = cnn3d1([49, 49, 49, 1])
model.load_weights('model/chp29')

#data = np.load('/media/izm/Entertain/fjj/classify/delted/dlt-LKDS-00813.mhd.npy')
data = np.load('data/dlt-LKDS-00990.mhd.npy')
lb = np.load('data/lb-LKDS-00990.mhd.npy')
res = do_predict(model, data)
print res,lb
exit()
'''
























#model = unet(inputs_shape=(512, 512, 512, 1))
unet = unet(inputs_shape=(512,512, 1))
unet.load_weights('unet-filt-0.58.mdl')
cnn3d = cnn3d1([49, 49, 49, 1])
cnn3d.load_weights('chp24')

#------------------------------------------------------------------------------------------
#read in the mhd file and csv
#train_path = 'dir05-08/'
train_path = 'part0/'
#train_path = '/media/izm/Entertain/fjj/train_subset00-04/'
file_list = glob(train_path + "*.mhd")
#file_list = ['/home/izm/work/fjj/data/train_subset00/LKDS-00066.mhd']

print file_list


#-----------------some parameters------------#
threshold = -320
thick = 32
batch = 4
zfile = open("result/wrong", 'w')
outpath = 'classify/'

cnt = 0
for fcount, img_file in enumerate(tqdm(file_list)):
    #if fcount < 107:
    #   continue
    #if mini_dddf.shape[0]>0:                      # some files may not have a nodule--skipping those 
    if True:                      # some files may not have a nodule--skipping those 
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        print img_array.shape
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        if img_array.shape[1] != 512 or img_array.shape[0] > 850:
            zfile.write(img_file+'\n')
            continue

        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        print "origin center = ", origin, "; scaling = ", spacing
        # go through all nodes (why just the biggest?)
        count = 0

        #### create the delt_scan from the original scans as u-net input
        img_cp = img_array.copy()
        mask_array = []
        for slc in img_cp:
            mask_array.append(get_segmented_lungs(slc))
        segmented = np.array(mask_array, np.int16)
        delt_scan = filt_vessels(segmented)
        name = img_file.split('/')[-1]
        
        
        input_scan = np.array([delt_scan]).transpose((1,2,3,0))
        
        unet_sgm = unet.predict(input_scan, batch_size = 3)
        unet_sgm = unet_sgm.transpose(3,0,1,2)
        unet_sgm = unet_sgm[0]
        print 'seperating: ',name
        org_cubes, segment_cubes, centers = seperate_nodule(img_array, delt_scan, unet_sgm)
	prob = do_predict(cnn3d, segment_cubes)
        np.save('result/prob/'+name, prob)
        np.save('result/center/'+name, centers)
        print 'finish file', name

zfile.close()

