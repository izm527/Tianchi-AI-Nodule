#segmented_ct_scan = segmented_ct_scan0[100:130]
#segmented_ct_scan = np_copy(segmented_ct_scan0)
selem = ball(3)
print selem
print segmented_ct_scan
#segmented_ct_scan[segmented_ct_scan < -330] = 0
#segmented_ct_scan[segmented_ct_scan >= -330] = 1

raw_input("bi closing")
binary = binary_closing(segmented_ct_scan)

label_scan = label(binary) 
print label_scan.shape
print label_scan
raw_input()

areas = [r.area for r in regionprops(label_scan)]
areas.sort(label_scan)
print len(areas)
rr =  regionprops(label_scan)

raw_input()

for r in rr:
    max_x, max_y, max_z = 0, 0, 0
    min_x, min_y, min_z = 1000, 1000, 1000
    
    rds = r.coords
    print len(rds)
    
    for c in rds:
        max_z = max(c[0], max_z)
        max_y = max(c[1], max_y)
        max_x = max(c[2], max_x)
        
        min_z = min(c[0], min_z)
        min_y = min(c[1], min_y)
        min_x = min(c[2], min_x)
    if (min_z == max_z or min_y == max_y or min_x == max_x or (len(areas) > 2 and r.area > areas[-3])):
        for c in rds:
            segmented_ct_scan[c[0], c[1], c[2]] = -1000
    else:
        index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))


