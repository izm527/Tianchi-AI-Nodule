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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import HTML
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


# functions
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)
####
# make the masking background from nodule info
####
def make_mask(center,diam,z,width,height,spacing,origin): #只显示结节
    '''
	Center : 圆的中心 px -- list of coordinates x,y,z
	diam : 圆的直径 px -- diameter
	widthXheight : pixel dim of image
	spacing = mm/px conversion rate np array x,y,z
	origin = x,y,z mm np.array
	z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width], np.uint8) 			#0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates
    #Defining the voxel range in which the nodule falls
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
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1
    return(mask)




threshold = -350
#####
# mask the ball space containing the whole nodule from the origin ct_scan
#####
def extract_nodule(ct_scan, masks, ll, hh):
	mask_scan = ct_scan.copy()
	for i in range(0, mask_scan.shape[0]):    
		if i >=  ll and i < hh:		#the slice is between the nodule's space
		    for j in range(512):
		        for k in range(512):
		            if masks[i-ll][j][k] == 0:                
		                mask_scan[i][j][k] = -1000		#outside the ball containing the nodule
		else:
		    mask_scan[i][:][:] = -1000					#outside the slices containing the nodule
	
	#mask_scan[mask_scan < threshold] = -1000

        #binarize the array
        unet_mask = mask_scan
        unet_mask[unet_mask > (threshold)] = 1
        unet_mask[unet_mask < 0 ] = 0

        return unet_mask


#####
# plot the scans in 3d mode
#####
def plot_3d(image, name, threshold=-350):
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
    face_color = [0.1, 0.1, 0.8]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()
    plt.clf(); plt.close()




#####
# remove the two biggest vessels in the lung
#####

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


def filt_vessels(segmented_ct_scan):
    print "filting vessels!"
    selem = ball(2)
    
    segmented_ct_scan0 = segmented_ct_scan.copy() 
    linking_scan = segmented_ct_scan0.copy()
    linking_scan[linking_scan >= threshold+10] = 1
    linking_scan[linking_scan < threshold+10] = 0

    binary = binary_closing(linking_scan, selem)
    label_scan = label(binary, neighbors=4)
    #label_scan = label(segmented_ct_scan, neighbors=4) 

    rr =  regionprops(label_scan)
    areas = [r.area for r in rr]
    areas.sort()
    print "areas: ", areas[-6:-1]

    vessels = segmented_ct_scan0.copy()
    vessels[:,:,:] = -1000
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
        if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-4]):
            for c in rds:
                vessels[c[0], c[1], c[2]] = segmented_ct_scan0[c[0], c[1], c[2]]
                segmented_ct_scan0[c[0], c[1], c[2]] = -1000
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))
    return segmented_ct_scan0, vessels




#------------------------------------------------------------------------------------------
#read in the mhd file and csv
#train_path = 'data/test_subset01/'
train_path = '/media/izm/Normal/train_subset00-04/'
file_list = glob(train_path + "*.mhd")
#file_list = glob(train_path + "*.mhd")
df_node = pd.read_csv('annotations.csv')
#out_path = '/media/sarah/003EE40D3EE3FA14/fjj-out/output/'
out_path = '/media/izm/Normal/FJJ/nodule-slcs-t/'

#4.1 将数据关联至csv
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna()

print file_list


#-----------------some parameters------------#
thick = 32
zfile = open("output/z-area-00-04.txt", 'w')
cnt = 0
skip = 0
z_splt = 128
for fcount, img_file in enumerate(tqdm(file_list)):
    if fcount < 0:
		continue
    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if mini_df.shape[0]>0:                      # some files may not have a nodule--skipping those 
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        print img_array.shape
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane

        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        print "origin center = ", origin, "; scaling = ", spacing


        # go through all nodes (why just the biggest?)
        count = 0

        #### create the delt_scan from the original scans as u-net input
        #img_cp = img_array.copy()
        #mask_array = []
        #for slc in img_array:
        #    mask_array.append(get_segmented_lungs(slc))
        #segmented = np.array(mask_array, np.int16)
        name = img_file.split('/')[-1]

        for node_idx, cur_row in mini_df.iterrows():			
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]

            center = np.array([node_x, node_y, node_z])   # nodule center
            v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
            print "v_center = ", v_center, "diam=", diam


            #clip to get the slices in the centre places
            thick = int(diam/spacing[2]/1.8) + 1
            ll = int(v_center[2]) - thick
            hh = int(v_center[2]) + thick
            rg = np.unique( np.arange(ll, hh).clip(0, num_z-1) )    #clip prevents going out of bounds in Z
            ll = rg[0]
            hh = rg[-1]+1
            masks = np.ndarray([hh-ll, height,width], dtype=np.uint8)
            #for i, i_z in enumerate(rg): 
            #    #print i, i_z
            #    #print "----------------------------------"
            #    mask = make_mask(center, diam, i_z*spacing[2]+origin[2],
            #                     width, height, spacing, origin)
            #    masks[i] = mask

            count += 1
            idx_str = '-'+str(count)
	    #res = extract_nodule(segmented, masks, ll, hh)
	    #if(np.max(res) == 0):
	    #    	print "----------------------No nodule! skip it!----------------------"
	    #    	skip += 1
	    #    	continue
            ##plot_3d(res, name + idx_str, 0)

            #clip to get the slices in the split_sizes of z
            z_idx = (int) (v_center[2] / z_splt)
            ll = z_idx*z_splt
            hh = ll + z_splt
            #if hh > num_z:
            #    print "Need to fill the blank!\n"
            #    blank = np.zeros([hh-num_z,512, 512], dtype = res.dtype)
            #    print "blank shape:", blank.shape
            #    clip_scan = np.concatenate((res[ll:num_z], blank), axis=0)
            #else:
            #    clip_scan = res[ll:hh]

            #plot_3d(clip_scan, name + idx_str, 0)
            print name+idx_str
            #print "clip_scan = ", clip_scan.shape
            print "ll---->hh:", ll, hh
	    #np.save(out_path+'zclip-'+name+idx_str, clip_scan)
            zfile.write(name+idx_str+' '+str(z_idx)+' '+str(v_center[0])+' '+str(v_center[1])+' '+str(v_center[2])+'\n')


zfile.close()
print "All skips = ", skip

