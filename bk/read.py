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

import matplotlib.pyplot as plt

import matplotlib.animation as animation
from IPython.display import HTML

    
import warnings #remove warnings
warnings.filterwarnings("ignore")





# functions
def make_mask(center,diam,z,width,height,spacing,origin): #只显示结节
    '''
Center : 圆的中心 px -- list of coordinates x,y,z
diam : 圆的直径 px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)



def matrix2int16(matrix):
    ''' 
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))


#
# Helper function to get rows in data frame associated 
# with each file



def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)


#
# The locations of the nodes
def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    """数据标准化"""
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image
    #---数据标准化
    
    
def set_window_width(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    """设置窗宽"""
    image[image > MAX_BOUND] = MAX_BOUND
    image[image < MIN_BOUND] = MIN_BOUND
    return image
    #---设置窗宽   








#-----------------------------------------------------------------------------------------------



#2.1 read in the csv file
luna_path = 'data/'
luna_subset_path = luna_path + 'sample/'
file_list = glob(luna_subset_path + "*.mhd")
df_node = pd.read_csv(luna_path + 'csv/train/' + 'annotations.csv')
out_path = 'output/'

#4.1 将数据关联至csv
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna()
#df_node.head(5)


print file_list
#print df_node

#raw_input('press entre to continue output')

'''
for fcount, img_file in enumerate(tqdm(file_list)):
    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if mini_df.shape[0]>0:                      # some files may not have a nodule--skipping those 
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane

        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)

        # go through all nodes (why just the biggest?)
        for node_idx, cur_row in mini_df.iterrows():       
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            # just keep 3 slices
            imgs = np.ndarray([3,height,width], dtype=np.float32)
            masks = np.ndarray([3,height,width], dtype=np.uint8)
            center = np.array([node_x, node_y, node_z])   # nodule center
            v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)

            #clip to get the slices in the centre places
            for i, i_z in enumerate(np.arange(int(v_center[2])-1,
                             int(v_center[2])+2).clip(0, num_z-1)): # clip prevents going out of bounds in Z
                print "----------------------------------"
                print i, i_z
                print "----------------------------------"
                mask = make_mask(center, diam, i_z*spacing[2]+origin[2],
                                 width, height, spacing, origin)
                masks[i] = mask
                imgs[i] = img_array[i_z]
                        
            #np.save(os.path.join(out_path,"images_%04d_%04d.npy" % (fcount, node_idx)),imgs)
            #np.save(os.path.join(out_path,"masks_%04d_%04d.npy" % (fcount, node_idx)),masks)
'''

def get_segmented_lungs(im, show=False):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if show == True:
        plot = plt.subplot()
	plot.imshow(im, cmap=plt.cm.bone)
	print "1"
	plt.show()
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -230
    if show == True:
        plot = plt.subplot()
        plot.imshow(binary, cmap=plt.cm.bone) 
	print "2"
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
    im[get_high_vals] = -1024
    if show == True:
 	plot = plt.subplot()
        plot.imshow(im, cmap=plt.cm.bone)
	plt.show()
       	

    return im


train_set_path = '/home/izm/work/fjj/data/train_subset00/'
file_list = glob(train_set_path + "*.mhd")
mask_out_path = '/home/izm/work/fjj/data/masked/msk_'

def mask_trainset(file_list):
    print file_list

    for fcount, img_file in enumerate(tqdm(file_list)):
        itk_img = sitk.ReadImage(img_file)
        nms = img_file.split('/')
        name = nms[-1]
        img_array = sitk.GetArrayFromImage(itk_img) 	# indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape        	#heightXwidth constitute the transverse plane
        mask_array = []
        for i in range(num_z):
            mask_array.append(get_segmented_lungs(img_array[i]))
    	np.save(mask_out_path + name, np.array(mask_array, 'short'))

mask_trainset(file_list)

	
'''
#print img_array.shape
#get_segmented_lungs(imgs[1], True)

reshp = np.zeros([512, 512, 3], 'float32')
for i in range(512):
    for j in range(512):
        reshp[i][j][0] = imgs[0][i][j]
        reshp[i][j][1] = imgs[1][i][j]
        reshp[i][j][2] = imgs[2][i][j]

plot = plt.subplot()
plot1 = plt.subplot()
gg = imgs[2]#reshp[:,:,0]
gg1 = reshp[:,:,1]
print gg1.shape, gg.shape
plot.imshow(gg, cmap=plt.cm.bone)
plot1.imshow(gg1, cmap=plt.cm.bone)
plt.show()
'''
