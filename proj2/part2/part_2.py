import numpy as np
from scipy import signal,misc
import scipy
import matplotlib.pyplot as plt
import cv2
import skimage.io as skio
from align_image_code import align_images

"""
PART 2.1
"""

#code from part 1
def apply_gaussian(im, ksize, sigma):
    g = cv2.getGaussianKernel(ksize, sigma)
    gaussian_filter = np.outer(g.T,g)
    blurred = signal.convolve2d(im, gaussian_filter, boundary='symm', mode='same')
    return blurred


def unsharp_mask(im):
    b,g,r = cv2.split(im)
    #r = im[:,:,0]
    #g = im[:,:,1]
    #b = im[:,:,2]

    KERNEL_SIZE = 5
    SIGMA = 3
    r_sharp = r - apply_gaussian(r, KERNEL_SIZE,SIGMA)
    g_sharp = g - apply_gaussian(g, KERNEL_SIZE,SIGMA)
    b_sharp = b - apply_gaussian(b, KERNEL_SIZE,SIGMA)
    return np.dstack([r_sharp, g_sharp, b_sharp])

def sharpen(im, alpha):
    return np.clip(unsharp_mask(im)+(alpha*im),0,255)    

#sharpen taj
taj = skio.imread("taj.jpg")
taj_sharp = sharpen(taj,1)
skio.imsave("taj_sharp.jpg", taj_sharp)

#plot the orig, sharpened taj
fig, (ax_orig, ax_sharp) = plt.subplots(1, 2, figsize=(10, 6))
ax_orig.imshow(taj)
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_sharp.imshow(taj_sharp.astype(int))
ax_sharp.set_title('Sharpened')
ax_sharp.set_axis_off()
plt.savefig('2_1_1.png')


#sharpen sample img
snacc = skio.imread("snacc.jpg")
snacc_sharp = sharpen(snacc,1)
skio.imsave("snacc_sharp.jpg", snacc_sharp)

#plot the orig, sharpened taj
fig, (ax_orig, ax_sharp) = plt.subplots(1, 2, figsize=(10, 6))
ax_orig.imshow(snacc)
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_sharp.imshow(snacc_sharp.astype(int))
ax_sharp.set_title('Sharpened')
ax_sharp.set_axis_off()
plt.savefig('2_1_2.png')
