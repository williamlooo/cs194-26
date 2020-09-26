import matplotlib.pyplot as plt
from align_image_code import align_images
import cv2 
import numpy as np
from scipy import signal
import skimage
import skimage.io as skio
# First load images

# high sf
#im1 = plt.imread('./DerekPicture.jpg')/255.

# low sf
#im2 = plt.imread('./nutmeg.jpg')/255

# Next align images (this code is provided, but may be improved)
#im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies
"""
Part 2.2
"""
def apply_gaussian(im, ksize, sigma):
    g = cv2.getGaussianKernel(ksize, sigma)
    gaussian_filter = np.outer(g.T,g)
    blurred = signal.convolve2d(im, gaussian_filter, boundary='symm', mode='same')
    return blurred

def apply_lowpass(im,sigma):
    KERNEL_SIZE = 13
    return apply_gaussian(im, KERNEL_SIZE,sigma)

def apply_highpass(im,sigma):
    KERNEL_SIZE = 9
    return  im - apply_gaussian(im, KERNEL_SIZE,sigma)

def hybrid_image(im1,im2,sigma1,sigma2):
    one = apply_lowpass(im1,sigma1)
    two = apply_highpass(im2,sigma2)
    return one,two

im1 = skio.imread("DerekPicture.jpg")
im2 = skio.imread("nutmeg.jpg")
im2_aligned, im1_aligned = align_images(im2, im1)

sigma1 = 11
sigma2 = 13
grey_im1_aligned = skimage.color.rgb2gray(im1_aligned)
grey_im2_aligned = skimage.color.rgb2gray(im2_aligned)
one,two = hybrid_image(grey_im1_aligned, grey_im2_aligned, sigma1, sigma2)
hybrid = one+two

#plot out input and result hybrids
fig, (ax_nutmeg, ax_derek, ax_hybrid) = plt.subplots(1, 3, figsize=(15, 6))
ax_nutmeg.imshow(im2)
ax_nutmeg.set_title('Nutmeg')
ax_nutmeg.set_axis_off()

ax_derek.imshow(im1)
ax_derek.set_title('Derek')
ax_derek.set_axis_off()

ax_hybrid.imshow(hybrid,cmap='gray')
ax_hybrid.set_title('Hybrid')
ax_hybrid.set_axis_off()
plt.savefig('2_2_1.png')
plt.clf()

#plot out the fourier transforms for the low and high pass results
fig, (ax_nutmeg, ax_derek) = plt.subplots(1, 2, figsize=(10, 6))
ax_nutmeg.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(skimage.color.rgb2gray(im1))))))
ax_nutmeg.set_title('Fourier Derek Lowpass')
ax_nutmeg.set_axis_off()

ax_derek.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(skimage.color.rgb2gray(im2))))))
ax_derek.set_title('Fourier Nutmeg Highpass')
ax_derek.set_axis_off()
plt.savefig('fourier_1.jpg')
plt.clf()

#plot out the fourier transforms for the low and high pass results
fig, (ax_nutmeg, ax_derek, ax_hybrid) = plt.subplots(1, 3, figsize=(15, 6))
ax_nutmeg.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(grey_im1_aligned)))))
ax_nutmeg.set_title('Fourier Derek Lowpass')
ax_nutmeg.set_axis_off()

ax_derek.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(grey_im2_aligned)))))
ax_derek.set_title('Fourier Nutmeg Highpass')
ax_derek.set_axis_off()

ax_hybrid.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid)))))
ax_hybrid.set_title('Fourier Hybrid')
ax_hybrid.set_axis_off()
plt.savefig('fourier_2.jpg')
plt.clf()


#HAWK+BOMBER
im1 = skio.imread("bomber.jpg")
im2 = skio.imread("hawk.png")

im2_aligned, im1_aligned = align_images(im2, im1)
sigma1 = 22
sigma2 = 11
grey_im1_aligned = skimage.color.rgb2gray(im1_aligned)
grey_im2_aligned = skimage.color.rgb2gray(im2_aligned)
one,two = hybrid_image(grey_im1_aligned, grey_im2_aligned, sigma1, sigma2)
hybrid = one+two

#plot out input and result hybrids
fig, (ax_one, ax_two, ax_hybrid) = plt.subplots(1, 3, figsize=(15, 6))
ax_one.imshow(im1)
ax_one.set_title('Bomber Jet')
ax_one.set_axis_off()

ax_two.imshow(im2)
ax_two.set_title('Hawk')
ax_two.set_axis_off()

ax_hybrid.imshow(hybrid,cmap='gray')
ax_hybrid.set_title('Hybrid Jet Hawk')
ax_hybrid.set_axis_off()
plt.savefig('2_2_2.png')
plt.clf()

#KIWI+KIWIBOT
im1 = skio.imread("kiwibot.jpg")
im2 = skio.imread("kiwi.jpg")
im2_aligned, im1_aligned = align_images(im2, im1)
sigma1 = 22
sigma2 = 11
grey_im1_aligned = skimage.color.rgb2gray(im1_aligned)
grey_im2_aligned = skimage.color.rgb2gray(im2_aligned)
one,two = hybrid_image(grey_im1_aligned, grey_im2_aligned, sigma1, sigma2)
hybrid = one+two

#plot out input and result hybrids
fig, (ax_one, ax_two, ax_hybrid) = plt.subplots(1, 3, figsize=(15, 6))
ax_one.imshow(im1)
ax_one.set_title('Kiwibot')
ax_one.set_axis_off()

ax_two.imshow(im2)
ax_two.set_title('a kiwi')
ax_two.set_axis_off()

ax_hybrid.imshow(hybrid,cmap='gray')
ax_hybrid.set_title('Kiwi kiwi')
ax_hybrid.set_axis_off()
plt.savefig('2_2_3.png')
plt.clf()

#HAPPY+SAD
im1 = skio.imread("will.jpg")
im2 = skio.imread("will2.jpg")
im2_aligned, im1_aligned = align_images(im2, im1)
sigma1 = 22
sigma2 = 11
grey_im1_aligned = skimage.color.rgb2gray(im1_aligned)
grey_im2_aligned = skimage.color.rgb2gray(im2_aligned)
one,two = hybrid_image(grey_im1_aligned, grey_im2_aligned, sigma1, sigma2)
hybrid = one+two

#plot out input and result hybrids
fig, (ax_one, ax_two, ax_hybrid) = plt.subplots(1, 3, figsize=(15, 6))
ax_one.imshow(im1)
ax_one.set_title('neutral')
ax_one.set_axis_off()

ax_two.imshow(im2)
ax_two.set_title('happy')
ax_two.set_axis_off()

ax_hybrid.imshow(hybrid,cmap='gray')
ax_hybrid.set_title('happysad')
ax_hybrid.set_axis_off()
plt.savefig('2_2_4.png')
plt.clf()


"""
Part 2.3
"""
def stack(hybrid,N):
    hybrid = skimage.color.rgb2gray(hybrid)
    gaussian_res = []
    laplacian_res = []
    gaussian_res.append(hybrid)

    for i in range(N-1):
        new_hybrid = apply_gaussian(hybrid,11,9)
        laplacian = hybrid-new_hybrid
        gaussian_res.append(hybrid)
        laplacian_res.append(laplacian)
        hybrid = new_hybrid
    laplacian_res.append(hybrid)
    return gaussian_res,laplacian_res

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
hybrid = skio.imread("dali.jpg")
gaussian_res,laplacian_res = stack(hybrid,N)

#save gaussian
fig, (ax_one, ax_two, ax_three, ax_four, ax_five) = plt.subplots(1, 5, figsize=(25, 6))
ax_one.imshow(gaussian_res[0],cmap='gray')
ax_one.set_axis_off()

ax_two.imshow(gaussian_res[1],cmap='gray')
ax_two.set_axis_off()

ax_three.imshow(gaussian_res[2],cmap='gray')
ax_three.set_axis_off()

ax_four.imshow(gaussian_res[3],cmap='gray')
ax_four.set_axis_off()

ax_five.imshow(gaussian_res[4],cmap='gray')
ax_five.set_axis_off()

plt.savefig('2_3_1.png')
plt.clf()

#save laplacian
fig, (ax_one, ax_two, ax_three, ax_four, ax_five) = plt.subplots(1, 5, figsize=(25, 6))
ax_one.imshow(laplacian_res[0],cmap='gray')
ax_one.set_axis_off()

ax_two.imshow(laplacian_res[1],cmap='gray')
ax_two.set_axis_off()

ax_three.imshow(laplacian_res[2],cmap='gray')
ax_three.set_axis_off()

ax_four.imshow(laplacian_res[3],cmap='gray')
ax_four.set_axis_off()

ax_five.imshow(laplacian_res[4],cmap='gray')
ax_five.set_axis_off()

plt.savefig('2_3_2.png')
plt.clf()

#do the same for photo from part 2
jet = skio.imread("hybrid_jet_final.jpg")
gaussian_res,laplacian_res = stack(jet,N)

#save gaussian
fig, (ax_one, ax_two, ax_three, ax_four, ax_five) = plt.subplots(1, 5, figsize=(25, 6))
ax_one.imshow(gaussian_res[0],cmap='gray')
ax_one.set_axis_off()

ax_two.imshow(gaussian_res[1],cmap='gray')
ax_two.set_axis_off()

ax_three.imshow(gaussian_res[2],cmap='gray')
ax_three.set_axis_off()

ax_four.imshow(gaussian_res[3],cmap='gray')
ax_four.set_axis_off()

ax_five.imshow(gaussian_res[4],cmap='gray')
ax_five.set_axis_off()

plt.savefig('2_3_3.png')
plt.clf()

#save laplacian
fig, (ax_one, ax_two, ax_three, ax_four, ax_five) = plt.subplots(1, 5, figsize=(25, 6))
ax_one.imshow(laplacian_res[0],cmap='gray')
ax_one.set_axis_off()

ax_two.imshow(laplacian_res[1],cmap='gray')
ax_two.set_axis_off()

ax_three.imshow(laplacian_res[2],cmap='gray')
ax_three.set_axis_off()

ax_four.imshow(laplacian_res[3],cmap='gray')
ax_four.set_axis_off()

ax_five.imshow(laplacian_res[4],cmap='gray')
ax_five.set_axis_off()

plt.savefig('2_3_4.png')
plt.clf()

"""
PART 2.4
"""
def blend(im1,im2,mask):
    gau_im1,lap_im1 = stack(im1,5)
    gau_im2,lap_im2 = stack(im2,5)
    gau_mask, lap_mask = stack(mask,5)

    partials = []
    for i in range(len(lap_im1)):
        mask_i = gau_mask[i]
        lap_1 = lap_im1[i]
        lap_2 = lap_im2[i]
        partials.append(mask_i*lap_1 + (1-mask_i)*lap_2) #per the laplacian blend eqn

    #sum together all layers
    res = partials[0]
    for i in range(1,len(partials)):
        res = res+partials[i]
    return res,partials

mask = skimage.color.rgb2gray(skio.imread("half_mask.jpg"))
apple = skio.imread("apple.jpeg")
orange = skio.imread("orange.jpeg")
orapple,_ = blend(orange,apple,mask)

#save orapple
fig, (ax_one, ax_two, ax_three, ax_four) = plt.subplots(1, 4, figsize=(20, 6))
ax_one.imshow(apple)
ax_one.set_axis_off()

ax_two.imshow(orange)
ax_two.set_axis_off()

ax_three.imshow(mask,cmap='gray')
ax_three.set_axis_off()

ax_four.imshow(orapple,cmap='gray')
ax_four.set_axis_off()

plt.savefig('2_4_1.png')
plt.clf()

#fall and door
mask = skimage.color.rgb2gray(skio.imread("rect_mask.jpg"))
fall = skio.imread("fall.jpg")
room = skio.imread("room.jpg")
res,partials = blend(fall,room,mask)

#save fall and door
fig, (ax_one, ax_two, ax_three, ax_four) = plt.subplots(1, 4, figsize=(20, 6))
ax_one.imshow(fall)
ax_one.set_axis_off()

ax_two.imshow(room)
ax_two.set_axis_off()

ax_three.imshow(mask,cmap='gray')
ax_three.set_axis_off()

ax_four.imshow(res,cmap='gray')
ax_four.set_axis_off()

plt.savefig('2_4_2.png')
plt.clf()

#save laplacian work for fall and door
fig, (ax_one, ax_two, ax_three, ax_four, ax_five) = plt.subplots(1, 5, figsize=(25, 6))
ax_one.imshow(partials[0],cmap='gray')
ax_one.set_axis_off()

ax_two.imshow(partials[1],cmap='gray')
ax_two.set_axis_off()

ax_three.imshow(partials[2],cmap='gray')
ax_three.set_axis_off()

ax_four.imshow(partials[3],cmap='gray')
ax_four.set_axis_off()

ax_five.imshow(partials[4],cmap='gray')
ax_five.set_axis_off()

plt.savefig('2_4_3.png')
plt.clf()


#kiwi and person
mask = skimage.color.rgb2gray(skio.imread("circle_filter.jpg"))
kiwi = skio.imread("kiwi.jpg")
guy = skio.imread("will_2.jpg")
res,_ = blend(guy,kiwi,mask)

#save kiwi and person
fig, (ax_one, ax_two, ax_three, ax_four) = plt.subplots(1, 4, figsize=(20, 6))
ax_one.imshow(kiwi)
ax_one.set_axis_off()

ax_two.imshow(guy)
ax_two.set_axis_off()

ax_three.imshow(mask,cmap='gray')
ax_three.set_axis_off()

ax_four.imshow(res,cmap='gray')
ax_four.set_axis_off()

plt.savefig('2_4_4.png')
plt.clf()

#fin
