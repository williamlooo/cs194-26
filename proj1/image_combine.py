# CS194-26 (CS294-26): Project 1 starter Python code
# By William Loo

import numpy as np
import os
import skimage as sk
from skimage import measure #used for ssim, comment out if import error.
import skimage.io as skio

# directory name
directory = "intake/"

#some loss metrics explored:
def structural_similarity(img1,img2):
    assert img1.shape == img2.shape
    c1, c2 = 1, 1
    #res = ((2*np.average(img1)*np.average(img2)+ c1) * (2*np.cov(img1,img2)+c2)) / ((np.average(img1)**2+np.average(img2)**2+c1)*(np.var(img1)**2 + np.var(img2)**2 + c2))
    res = measure.compare_ssim(img1, img2)
    return res

def zero_normalized_cross_correlation(img1, img2):
    assert img1.shape == img2.shape
    return np.average(np.multiply(img1 - np.average(img1), img2 - np.average(img2))) / (np.std(img1) * np.std(img2))

def sum_squared_differences(img1, img2):
    assert img1.shape == img2.shape
    return np.sum(np.sum((img1-img2)**2))


# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
def align(a,b):
    max_acc = float("-inf")
    min_shift = (0,0)
    best_scale = 0.5

    #initial search range is intentionally large
    search_range_x = (-200,200)
    search_range_y = (-200,200) 

    scales = (0.125,0.25,0.5,1.0)

    #smaller images do not need pyramid, and can be well aligned using 0.5 scale.
    if len(a) < 1000:
        search_range_x = (-20,20)
        search_range_y = (-20,20) 
        scales = [0.5]

    for scale in scales:
        resized_a = sk.transform.rescale(a,scale)
        resized_b = sk.transform.rescale(b,scale)
        
        #cut 20% off from each side to narrow down search space
        a_side_border_width = len(resized_a[0]) // 5
        a_top_bottom_border_width = len(resized_a) // 5
        b_side_border_width = len(resized_b[0]) // 5
        b_top_bottom_border_width = len(resized_b) // 5
        resized_a = resized_a[a_top_bottom_border_width:-a_top_bottom_border_width,a_side_border_width:-a_side_border_width]
        resized_b = resized_b[b_top_bottom_border_width:-b_top_bottom_border_width,b_side_border_width:-b_side_border_width]

        #searching and matching
        for i in range(int(search_range_x[0]*scale), int(search_range_x[1]*scale), 1):
            for j in range(int(search_range_y[0]*scale),int(search_range_y[1]*scale),1):
                shifted_a = np.roll(np.roll(resized_a,i,1),j,0)
                #acc = zero_normalized_cross_correlation(shifted_a, resized_b)
                acc = structural_similarity(shifted_a, resized_b)
                #print("shift corr: {}, out of {}, for shift {}, {}".format(acc, max_acc, i,j))
                if acc > max_acc:
                    min_shift = (i,j) #hor, vert
                    max_acc = acc
                    best_scale = scale
                    #print("found optimal scale at {} with shift {}, {}".format(scale, i, j))

        #update search range
        c_x = int((min_shift[0])*(1/scale))
        c_y = int((min_shift[1])*(1/scale))
        search_range_x = (c_x-int(1/scale),c_x+int(1/scale))
        search_range_y = (c_y-int(1/scale),c_y+int(1/scale))

    h = int((min_shift[0])*(1/best_scale))
    v = int((min_shift[1])*(1/best_scale))
    print((h,v))
    return np.roll(np.roll(a,h,1),v,0)


for imname in os.listdir(directory):
    if imname.endswith(".jpg") or imname.endswith(".tif"):
        print("reading {}..".format(imname))
        # read in the image
        im = skio.imread("{}/{}".format(directory,imname))

        # convert to double (might want to do this later on to save memory)    
        im = sk.img_as_float(im)
            
        # compute the height of each part (just 1/3 of total)
        height = np.floor(im.shape[0] / 3.0).astype(np.int)

        # separate color channels
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        ar = align(r, b)
        print("done shifting r")
        ag = align(g, b)
        print("done shifting g") 
        # create a color image
        im_out = np.dstack([ar, ag, b])

        """
        #shifted variant to help with emir.tif
        ar = align(r, g)
        print("done shifting red layer..")
        ab = align(b, g)
        print("done shifting blue layer..") 
        # create a color image
        im_out = np.dstack([ar, g, ab])
        """

        #demo use 
        #im_out = np.dstack([r, [[0 for i in range(len(r[0]))] for j in range(len(r))], [[0 for i in range(len(r[0]))] for j in range(len(r))]])
        #im_out = np.dstack([[[0 for i in range(len(r[0]))] for j in range(len(r))], g, [[0 for i in range(len(r[0]))] for j in range(len(r))]])
        #im_out = np.dstack([[[0 for i in range(len(r[0]))] for j in range(len(r))], [[0 for i in range(len(r[0]))] for j in range(len(r))],b])

        # save the image
        if imname.endswith(".tif"):
            imname = imname.split(".")[0]+".jpg"
        fname = 'out_path/{}'.format(imname)
        skio.imsave(fname, im_out)
