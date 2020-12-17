import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy import ndimage

def read_imgs(folder):
    """
    returns a list of images
    """
    #if os.path.exists("{}.p".format(folder)):
    #    imgs = pickle.load(open("{}.p".format(folder),"rb"))
    #else:
    if True:
        imgs = []
        for filename in os.listdir("inputs/{}".format(folder)):
            parts = filename.split("_")
            im = plt.imread("inputs/{}/{}".format(folder,filename))
            imgs.append([copy.deepcopy([int(parts[1]),int(parts[2])]), copy.deepcopy(im),filename])
        #pickle.dump(imgs,open("{}.p".format(folder),"wb"))

    return imgs

def refocus(imgs,scale,center=8):
    """
    refocusing based on x,y shift values indicated from filename
    center value used to support different grid sizes
    """
    print("refocus at {}".format(scale))
    shifted_imgs = []
    for item in imgs:
        y_shift = int((center-item[0][0])*scale)
        x_shift = int((center-item[0][1])*scale)
        img = item[1]
        shifted = ndimage.shift(img, (y_shift, -x_shift, 0), mode="nearest", order=0, cval=0)
        shifted_imgs.append(shifted)
    res = np.mean(shifted_imgs,axis=0)
    return res

def adjust_aperture(imgs, radius,scale=0,center=8):
    """
    aperture adjustment by selecting the images within RADIUS pixels away from center
    center value to support different grid sizes
    scaling value used for centering.
    """
    print("adjusting aperture with radius: {}".format(radius))
    selected_imgs = []
    for item in imgs:
        shift = item[0]
        if abs(center-shift[0]) <= radius and abs(center-shift[1]) <= radius:
            x_shift = int((center-item[0][0])*scale)
            y_shift = int((center-item[0][1])*scale)
            img = item[1]
            shifted = ndimage.shift(img, (y_shift, -x_shift, 0), mode="nearest", order=0, cval=0)
            selected_imgs.append(shifted)
    res = np.mean(selected_imgs, axis=0)
    return res

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='lightfield depth refocusing and aperture simulation.')
    parser.add_argument("--folder", default="chess")
    args = parser.parse_args()

    folder = args.folder #chess, beans, desk, or flower
    print("selected: {}".format(folder))
    scale_range_config = {"chess":[0,3.3],"beans":[-5,1.1],"flower":[-2,3.1],"desk":[-2,2.1]}
    if folder not in scale_range_config:
        raise ValueError("Please select from chess, beans, desk, or flower.")
    scale_range = scale_range_config[folder]
    imgs = read_imgs(folder)
    
    #refocus
    count = 0
    if not os.path.exists("refocus_{}".format(folder)):
        os.makedirs("refocus_{}".format(folder))
    start,end = scale_range[0], scale_range[1]
    for s in np.arange(start,end,0.1):
        res = refocus(imgs,s)
        plt.imsave("refocus_{}/res_{}.png".format(folder,count),res)
        count+=1
    
    #aperture 
    centered_apertures = {"chess":1.5,"beans":-1.3,"flower":2,"desk":0} #TODO: tune these values
    if not os.path.exists("aperture_{}".format(folder)):
        os.makedirs("aperture_{}".format(folder))

    count = 0
    if folder == "desk": #desk data only has 9 images
        for aperture in np.arange(1,16,7):
            res = adjust_aperture(imgs, aperture)
            plt.imsave("aperture_{}/res_{}.png".format(folder,count),res)
            count+=1
    else:
        for aperture in np.arange(1,7): #otherwise use standard dataset of 289 images
            #res = adjust_aperture(imgs, aperture,scale=centered_apertures[folder])
            res = adjust_aperture(imgs, aperture)
            plt.imsave("aperture_{}/res_{}.png".format(folder,count),res)
            count+=1
    print("done.")
