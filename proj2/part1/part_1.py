import numpy as np
from scipy import signal,misc
import scipy
import matplotlib.pyplot as plt
import cv2

def get_fd_operator():
    fd_x = np.array([[1,-1]])
    fd_y = np.transpose(fd_x)
    return fd_x,fd_y

def fd_gradient_x(im):
    fd_x,fd_y = get_fd_operator()
    return signal.convolve2d(im, fd_x, boundary='symm', mode='same')    

def fd_gradient_y(im):
    fd_x,fd_y = get_fd_operator()
    return signal.convolve2d(im, fd_y, boundary='symm', mode='same')  

def gradient_magnitude(grad_x,grad_y):
    return np.sqrt(grad_x**2+grad_y**2)

def binarize(im, th):
    return (im > th) * 255

"""
PART 1.1
"""
cameraman = cv2.imread('cameraman.png',0)

grad_x = fd_gradient_x(cameraman)
grad_y = fd_gradient_y(cameraman)

#save the two derviative images
fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(10, 6))
ax_x.imshow(grad_x, cmap='gray')
ax_x.set_title('partial wrt x')
ax_x.set_axis_off()

ax_y.imshow(grad_y, cmap='gray')
ax_y.set_title('partial wrt y')
ax_y.set_axis_off()
plt.savefig('1_1_1.png')

BINARIZE_THRESHOLD = 50
grad = gradient_magnitude(grad_x,grad_y)
bin_grad = binarize(grad, BINARIZE_THRESHOLD)

#save original, gradient mag, and gradient binarized
fig, (ax_orig, ax_mag,ax_bin) = plt.subplots(1, 3, figsize=(15, 6))
ax_orig.imshow(cameraman, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_mag.imshow(grad, cmap='gray')
ax_mag.set_title('Gradient magnitude')
ax_mag.set_axis_off()

ax_bin.imshow(bin_grad, cmap='gray')
ax_bin.set_title('Gradient binarize')
ax_bin.set_axis_off()
plt.savefig('1_1_2.png')

"""
PART 1.2
"""
def apply_gaussian(im, ksize, sigma):
    fd_x,fd_y = get_fd_operator()
    g = cv2.getGaussianKernel(ksize, sigma)
    gaussian_filter = np.outer(g.T,g)
    dog_x = signal.convolve2d(gaussian_filter,fd_x, boundary='symm', mode='same')
    dog_y = signal.convolve2d(gaussian_filter,fd_y, boundary='symm', mode='same')
    blurred = signal.convolve2d(im, gaussian_filter, boundary='symm', mode='same')
    return dog_x, dog_y, blurred

BINARIZE_THRESHOLD_GAUSSIAN = 15
KERNEL_SIZE = 5
SIGMA = 7
dog_x,dog_y,blurred = apply_gaussian(cameraman,KERNEL_SIZE,SIGMA)
blur_grad_x = fd_gradient_x(blurred)
blur_grad_y = fd_gradient_y(blurred)
grad_blurred = gradient_magnitude(blur_grad_x,blur_grad_y)
bin_blurred = binarize(grad_blurred, BINARIZE_THRESHOLD_GAUSSIAN)

#plot the gradients
fig, (ax_orig, ax_blur) = plt.subplots(1, 2, figsize=(10, 6))
ax_orig.imshow(dog_x, cmap='gray')
ax_orig.set_title('DoG X')
ax_orig.set_axis_off()

ax_blur.imshow(dog_y, cmap='gray')
ax_blur.set_title('DoG Y')
ax_blur.set_axis_off()
plt.savefig('1_2_1.png')

#plot the orig, blurred, gradient, and binarized
fig, (ax_orig, ax_blur, ax_mag, ax_bin) = plt.subplots(1, 4, figsize=(20, 6))
ax_orig.imshow(cameraman, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_blur.imshow(blurred, cmap='gray')
ax_blur.set_title('Blurred')
ax_blur.set_axis_off()

ax_mag.imshow(grad_blurred, cmap='gray')
ax_mag.set_title('Gradient magnitude')
ax_mag.set_axis_off()

ax_bin.imshow(bin_blurred, cmap='gray')
ax_bin.set_title('Gradient binarize')
ax_bin.set_axis_off()
plt.savefig('1_2_2.png')

dog_x,dog_y,_ = apply_gaussian(cameraman,KERNEL_SIZE,SIGMA)
confirm_blur_grad_x = signal.convolve2d(cameraman, dog_x, boundary='symm', mode='same')
confirm_blur_grad_y = signal.convolve2d(cameraman, dog_y, boundary='symm', mode='same')

confirm_grad_blurred = gradient_magnitude(blur_grad_x,blur_grad_y)
confirm_bin_blurred = binarize(grad_blurred, BINARIZE_THRESHOLD_GAUSSIAN)

#plot the orig, blurred, gradient, and binarized using combined dog filter, and verify correctness
fig, (ax_orig, ax_blur, ax_mag, ax_bin) = plt.subplots(1, 4, figsize=(20, 6))
ax_orig.imshow(cameraman, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_blur.imshow(blurred, cmap='gray')
ax_blur.set_title('Blurred')
ax_blur.set_axis_off()

ax_mag.imshow(confirm_grad_blurred, cmap='gray')
ax_mag.set_title('Gradient magnitude with DoG filter')
ax_mag.set_axis_off()

ax_bin.imshow(confirm_bin_blurred, cmap='gray')
ax_bin.set_title('Gradient binarize with DoG filter')
ax_bin.set_axis_off()
plt.savefig('1_2_3.png')


"""
PART 1.3
"""
def gradient_angle(im):
    """
    Calculate gradient angle using arctan(grady/gradx)
    """
    KERNEL_SIZE = 5
    SIGMA = 5
    _,_, blurred = apply_gaussian(im,KERNEL_SIZE,SIGMA)
    grad_x = fd_gradient_x(blurred)
    grad_y = fd_gradient_y(blurred)   
    return np.degrees(np.arctan2(grad_y,grad_x))

def sum_hor_vert_edge(n, bins):
    """
    Sum up horizontal and vertical edges, within one bin width of -180,-90,0,90,180
    """
    ret = 0
    #0 deg
    zero_left_idx = np.where(bins == -4)[0][0]
    zero_right_idx = np.where(bins == 4)[0][0]
    for i in range(zero_left_idx,zero_right_idx+1):
        ret+=n[i]
    #90 deg
    p90_left_idx = np.where(bins == 88)[0][0]
    p90_right_idx = np.where(bins == 92)[0][0]
    for i in range(p90_left_idx,p90_right_idx+1):
        ret+=n[i]
    #-90 deg
    n90_left_idx = np.where(bins == -92)[0][0]
    n90_right_idx = np.where(bins == -88)[0][0]
    for i in range(n90_left_idx,n90_right_idx+1):
        ret+=n[i]
    #-180 deg
    n180_left_idx = np.where(bins == -180)[0][0]
    n180_right_idx = np.where(bins == -176)[0][0]
    for i in range(n180_left_idx,n180_right_idx+1):
        ret+=n[i]
    #180 deg
    p180_left_idx = np.where(bins == 176)[0][0]
    p180_right_idx = np.where(bins == 180)[0][0]
    for i in range(p180_left_idx,p180_right_idx):
        ret+=n[i]
    return ret

def find_optimal_tilt(im, out_hist_path):
    print(out_hist_path)
    opt_angle = 0
    max_edge_cnt = 0
    for a in np.arange(-10,10,1):
        rot_im = scipy.ndimage.interpolation.rotate(im, a)
        #cut 10% off from each side
        side_border_width = len(rot_im[0]) // 5
        top_bottom_border_width = len(rot_im) // 5
        cropped_im = rot_im[top_bottom_border_width:-top_bottom_border_width,side_border_width:-side_border_width]
        angles = gradient_angle(cropped_im)
        fig = plt.figure()
        plt.title("Rotated")
        (n, bins, patches) = plt.hist(np.ndarray.flatten(angles),np.arange(-180, 181, 2))
        plt.xticks(np.arange(-180, 181, 45))

        edge_cnt = sum_hor_vert_edge(n,bins)
        if edge_cnt > max_edge_cnt:
            max_edge_cnt = edge_cnt
            opt_angle = a
            #print("better: {}".format(opt_angle))
            plt.savefig(out_hist_path)
        plt.close()
    print("optimal angle found at: {}".format(opt_angle))
    return opt_angle

facade = cv2.imread('facade.jpg',0)
color_facade = cv2.imread('facade.jpg')
color_facade = cv2.cvtColor(color_facade, cv2.COLOR_BGR2RGB)
#plot original
hist_fig = plt.figure()
angles = gradient_angle(facade)
ax = plt.hist(np.ndarray.flatten(angles),np.arange(-180, 181, 2))
plt.xticks(np.arange(-180, 181, 45))
plt.title("Original")
plt.savefig("hist_facade_orig.jpg")
opt_angle = find_optimal_tilt(facade, "hist_facade_best.jpg")
best_rot = scipy.ndimage.interpolation.rotate(color_facade, float(opt_angle))

#plot the orig, rotated, and histograms
fig, (ax_orig, ax_blur, ) = plt.subplots(1, 2, figsize=(10, 6))
ax_orig.imshow(color_facade)
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_blur.imshow(best_rot)
ax_blur.set_title('Rotated')
ax_blur.set_axis_off()
plt.savefig('1_3_1.png')

"""
BOARD.JPG 
"""
board = cv2.imread('board.jpg',0)
color_board = cv2.imread('board.jpg')
color_board = cv2.cvtColor(color_board, cv2.COLOR_BGR2RGB)
#plot original
hist_fig = plt.figure()
angles = gradient_angle(board)
ax = plt.hist(np.ndarray.flatten(angles),np.arange(-180, 181, 2))
plt.xticks(np.arange(-180, 181, 45))
plt.title("Original")
plt.savefig("hist_board_orig.jpg")
opt_angle = find_optimal_tilt(board, "hist_board_best.jpg")
best_rot = scipy.ndimage.interpolation.rotate(color_board, float(opt_angle))

#plot the orig, rotated, and histograms
fig, (ax_orig, ax_blur, ) = plt.subplots(1, 2, figsize=(10, 6))
ax_orig.imshow(color_board)
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_blur.imshow(best_rot)
ax_blur.set_title('Rotated')
ax_blur.set_axis_off()
plt.savefig('1_3_2.png')

"""
CITY.JPG 
"""
city = cv2.imread('city.jpg',0)
color_city = cv2.imread('city.jpg')
color_city = cv2.cvtColor(color_city, cv2.COLOR_BGR2RGB)
#plot original
hist_fig = plt.figure()
angles = gradient_angle(city)
ax = plt.hist(np.ndarray.flatten(angles),np.arange(-180, 181, 2))
plt.xticks(np.arange(-180, 181, 45))
plt.title("Original")
plt.savefig("hist_city_orig.jpg")
opt_angle = find_optimal_tilt(city, "hist_city_best.jpg")
best_rot = scipy.ndimage.interpolation.rotate(color_city, float(opt_angle))

#plot the orig, rotated, and histograms
fig, (ax_orig, ax_blur, ) = plt.subplots(1, 2, figsize=(10, 6))
ax_orig.imshow(color_city)
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_blur.imshow(best_rot)
ax_blur.set_title('Rotated')
ax_blur.set_axis_off()
plt.savefig('1_3_3.png')


"""
TREAT.JPEG 
"""
treat = cv2.imread('treat.jpeg',0)
color_treat = cv2.imread('treat.jpeg')
color_treat = cv2.cvtColor(color_treat, cv2.COLOR_BGR2RGB)
#plot original
hist_fig = plt.figure()
angles = gradient_angle(treat)
ax = plt.hist(np.ndarray.flatten(angles),np.arange(-180, 181, 2))
plt.xticks(np.arange(-180, 181, 45))
plt.title("Original")
plt.savefig("hist_treat_orig.jpg")
opt_angle = find_optimal_tilt(treat, "hist_treat_best.jpg")
best_rot = scipy.ndimage.interpolation.rotate(color_treat, float(opt_angle))

#plot the orig, rotated, and histograms
fig, (ax_orig, ax_blur, ) = plt.subplots(1, 2, figsize=(10, 6))
ax_orig.imshow(color_treat, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_blur.imshow(best_rot, cmap='gray')
ax_blur.set_title('Rotated')
ax_blur.set_axis_off()
plt.savefig('1_3_4.png')

#PART 2 CODE is inside folder part2
