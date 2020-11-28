#!/usr/bin/env python3

import argparse
import harris
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from skimage import io,transform
import skimage.draw as draw

def getPoints(im1, im2,rectify=False,num=8):
    print('Please select {} points in each image for alignment.'.format(num))
    points_1 = []
    points_2 = []
    if not rectify:
        plt.imshow(im1)
        points_1 = np.asarray(plt.ginput(num,timeout=0))
        plt.close()

    plt.imshow(im2)
    points_2 = np.asarray(plt.ginput(num,timeout=0))
    plt.close()

    if rectify:
        """
        rectify, p1 is square
        assumption is p2 is rectangular image in landscape form
        max_y = int(max(points_2,key=lambda p: p[1])[1])
        min_y = int(min(points_2, key=lambda p: p[1])[1])
        """

        #height = max_y-min_y
        height,width,ch = im2.shape
        ul = [0.,0.]
        ur = [height,0.]
        bl = [0.,height]
        br = [height,height]
        points_1 = [ul,ur,bl,br]
    
    print("im1:\n {}".format(points_1))
    print("im2:\n {}".format(points_2))
    return points_1,points_2

def computeH(im1_pts,im2_pts):
    """
    returns homography im2 to im1 given two sets of points
    """

    A = []
    for i in range(len(im1_pts)):
        p1 = im1_pts[i]
        p2 = im2_pts[i]
        im1x = p1[0]
        im1y = p1[1]
        im2x = p2[0]
        im2y = p2[1]

        A.append([im2x,im2y,1,0,0,0,-(im2x*im1x),-(im2y*im1x)])
        A.append([0,0,0,im2x,im2y,1,-(im2x*im1y),-(im2y*im1y)])

    A = np.array(A)

    b = []
    for p in im1_pts:
        b.extend(p[0:2])
    b = np.array([b]).T
    H = np.linalg.lstsq(A,b,rcond=-1)[0]

    return np.matrix([[H[0].item(), H[1].item(), H[2].item()],
                      [H[3].item(), H[4].item(), H[5].item()],
                       [H[6].item(), H[7].item(), 1.]])

def warpImage(im1,im2, H, rectify=False):
    """
    returns warped image im2 into im1 according to homography
    """
    #height,width,ch = im1.shape
    height,width,ch = im2.shape

    #CALCULATE BOUNDS
    ll = [0, height, 1]
    lr = [width, height, 1]
    ul = [0, 0, 1]
    ur = [width, 0, 1]

    bounds = [ll,lr,ul,ur]
    print("TRANSFORMED BOUNDS")
    transformed_bounds = np.squeeze(np.asarray([H.dot(p) for p in bounds]))

    transformed_bounds = [p / p[2] for p in transformed_bounds]
    
    print(transformed_bounds)

    max_x = int(max(transformed_bounds,key= lambda p: p[0])[0])
    max_y = int(max(transformed_bounds,key= lambda p: p[1])[1])
    min_x = int(min(transformed_bounds, key= lambda p: p[0])[0])
    min_y = int(min(transformed_bounds, key= lambda p: p[1])[1])

    print(max_x,max_y,min_x,min_y)

    max_x_all = max(max_x, im1.shape[1], im2.shape[1])
    max_y_all = max(max_y, im1.shape[0], im2.shape[0])

    move_x = abs(min(0,min_x))
    move_y = abs(min(0,min_y))

    canvas_height = max_y_all+abs(min_y)
    canvas_width = max_x_all+abs(min_x)

    canvas = np.zeros((canvas_height+1, canvas_width+1, 3))

    #DRAW POLYGON COMPUTE NEW IMAGE SIZE
    o_cc,o_rr = draw.polygon([0,canvas_width,canvas_width,0],[0,0,canvas_height,canvas_height])

    #generate coordinates and perform transformation
    canvas_coords = np.vstack([o_cc,o_rr, np.ones(len(o_rr))]) #(cc,rr,1) for len(o_rr)
    transformed = np.linalg.inv(H).dot(canvas_coords)
    #transformed = H.dot(canvas_coords)
    new_cc, new_rr, new_w = transformed
    new_cc = (new_cc / new_w).astype(int)
    new_rr = (new_rr / new_w).astype(int)
    after = [new_cc,new_rr,new_w]

    #now filter out-of-range values (kinda hardcode oops)
    valid_indices = np.where((new_rr >= 0) & (new_rr < im2.shape[0]) & (new_cc >= 0) & (new_cc < im2.shape[1]))

    new_cc = new_cc[valid_indices].astype(int)
    new_rr = new_rr[valid_indices].astype(int)
    o_cc = np.array([o_cc])[valid_indices].astype(int)+move_x
    o_rr = np.array([o_rr])[valid_indices].astype(int)+move_y

    #paaste over the new image
    canvas[o_rr, o_cc] = im2[new_rr, new_cc]

    print("done pasting")
    if not rectify:
        """
        # now for blending.
        #https://matplotlib.org/3.2.1/gallery/images_contours_and_fields/image_transparency_blend.html
        #strategy is to make first half-ish opaque, and last half transparent
        """
        half = int(width*0.5) #empirical, could be adjusted
        no_change_half = np.ones(half)
        gradient = np.linspace(1, 0, half)
        alpha = np.append(no_change_half,gradient)
        
        temp = []
        [temp.append(alpha) for _ in range(height)]
        temp = np.array(temp)
          
        alpha = temp.reshape((height, width, 1))
        transparent_im1 = im1*alpha

        range_x = im1.shape[1]+move_x
        range_y = im1.shape[0]+move_y
        
        print(move_y,range_y,move_x,range_x)

        #copy over gradient
        canvas[move_y:range_y,move_x:range_x] = (alpha)*transparent_im1 + (1-alpha)*canvas[move_y:range_y, move_x:range_x]
        
    plt.figure(dpi=240) #change dpi from 100 to 240 for higher quality
    plt.imshow(canvas.astype(np.uint8))
    plt.savefig("canvas.jpg",bbox_inches='tight')
    print('done warping')

def adaptive_nonmaximal_suppression(points, H, th, goal_num=250):
    """
    computes the suppression radius for each point 
    sorts the points by radius, and truncates the list 
    based on the requested number of interest points.
    """
    radiuses = {}

    for center_point in points:
        interest_points = []
        H_center = H[center_point[0],center_point[1]]
        for p in points:
            H_point = H[p[0],p[1]]
            interest_points.append(p) if H_center/H_point < th else None

        if interest_points:
            interest_points = np.array(interest_points)
            harris_dists = harris.dist2(np.array([center_point]),interest_points)
            radiuses[center_point] = min(harris_dists[0])

    selected_pts = []
    radiuses = sorted([(v, k) for (k, v) in radiuses.items()],reverse=True)
    for i in range(goal_num):
        selected_pts.append((radiuses[i][1][1],radiuses[i][1][0]))
    return selected_pts

def find_descriptors(im, pts,patch_size=40):
    res = {}
    for p in pts: #start top left and take 40x40 patches
        offset_col = int(p[0]-patch_size//2)
        offset_row = int(p[1]-patch_size//2)

        sample = np.array(im[offset_row:(offset_row+patch_size),offset_col:(offset_col+patch_size)])

        downsample = transform.resize(sample, (8, 8), anti_aliasing=False) #blur downsample to axis-aligned 8x8
        normed = (downsample-np.mean(downsample))/np.std(downsample) #normalize
        """
        #UNCOMMENT for viewing the descriptors
        plt.figure()
        plt.imshow(normed,cmap='gray')
        plt.savefig("feature_desc.jpg")
        """
        flat = normed.flatten().reshape(1,64)
        res[p] = flat
    return res

def match_features(descriptor_one, descriptor_two,th=0.5):
    """
    matches between all points and caaluclates
    the ratio between the two closest neighbors
    discards outliers if necessary
    """
    res = {}
    for point_one, desc_one in descriptor_one.items():
        distances_from_one = {}
        for point_two, desc_two in descriptor_two.items():
            dist = harris.dist2(desc_one, desc_two)
            distances_from_one[point_two] = dist[0][0]

        distances_from_one_arr = sorted((v, k) for (k, v) in distances_from_one.items()) #smallest
        #point one matching
        ratio = distances_from_one_arr[0][0]/distances_from_one_arr[1][0]
        if ratio < th: #outlier rejection
            res[point_one] = distances_from_one_arr[0][1] #replace with k
    return res

def generate_ransac_points(matched_pts,err_th=0.5,n=100):
    """
    samples 4 points for n times and calculates loss.
    """
    NUM_SAMPLES = 4
    final_pts = {}
    final_cts = 0
    for _ in range(n):
        res = {}
        one = list(matched_pts.keys())
        #print("one: {}".format(one))
        two = list(matched_pts.values())
        #print("two: {}".format(two))

        sample_indices = random.sample(range(len(one)), NUM_SAMPLES) #sample w/o replacement
        sample_one = []
        sample_two = []
        for i in range(NUM_SAMPLES):
            sample_one.append(one[sample_indices[i]])
            sample_two.append(two[sample_indices[i]])

        H = computeH(np.asarray(sample_two),np.asarray(sample_one)) #one to two
        p_prime = np.array(two).T

        #compute on ALL points
        out = H.dot(np.transpose(np.hstack((one, np.ones((len(one),1)))))) 
        compare = np.zeros(out.shape)
        #rescale based on homo output
        for i in range(3):
            compare[i,:]=out[i,:] / out[2, :]
        compare = np.delete(compare,2,0)
        #calculate loss
        loss_x = (compare[0].T - p_prime[0]) ** 2
        loss_y = (compare[1].T - p_prime[1]) ** 2
        squared_err = np.sqrt(loss_x+loss_y)
        
        #count inliers
        for i in range(len(squared_err)):
            if squared_err[i] < err_th:
                res[one[i]]=two[i]

        #addd if greaater inliers, use these to calculate correpondence
        if len(res) > len(final_pts):
            final_pts = res.copy()

    return final_pts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='panoramas using homographies.')
    parser.add_argument('--manual', action='store_true',
                        help='to use manual select points or not (default: %(default)s)')
    parser.add_argument('--rectify', action='store_true',
                        help='to rectify or not to rectify (default: %(default)s)')
    args = parser.parse_args()

    #load images, set image names here
    IM_1_NAME = "street_1.jpg"
    IM_2_NAME = "street_2.jpg"

    one = io.imread(IM_1_NAME)
    two = io.imread(IM_2_NAME)

    if not args.manual and args.rectify:
        """
        rectify overrides manual flag
        """
        args.manual = True

    if args.rectify:
        print("rectify")
        p_one, p_two = getPoints(one,two,rectify=args.rectify,num=4)
        H = computeH(np.asarray(p_one),np.asarray(p_two))
        warpImage(one,two,H,rectify=args.rectify)
        exit(0)

    elif not args.manual:
        #autostiching!
        one_bw = io.imread(IM_1_NAME, as_gray=True)
        two_bw = io.imread(IM_2_NAME, as_gray=True)

        H1, coords_1 = harris.get_harris_corners(one_bw)
        H2, coords_2 = harris.get_harris_corners(two_bw)
        plt.figure()
        plt.imshow(one)
        plt.scatter(coords_1[1], coords_1[0],s=1,color='orange')
        plt.savefig('harris.jpg')

        plt.figure()
        plt.imshow(two)
        plt.scatter(coords_2[1], coords_2[0],s=1,color='orange')
        plt.savefig('harris2.jpg')

        points_1 = list(zip(coords_1[0],coords_1[1]))
        suppressed_points_1 = adaptive_nonmaximal_suppression(points_1,H1,0.9,goal_num=500)
        plt.figure()
        plt.imshow(one)
        plt.scatter([p[0] for p in suppressed_points_1], [p[1] for p in suppressed_points_1], s = 1,c='orange')
        plt.savefig("anms1.jpg")

        points_2 = list(zip(coords_2[0],coords_2[1]))
        suppressed_points_2 = adaptive_nonmaximal_suppression(points_2,H2,0.9,goal_num=500)
        plt.figure()
        plt.imshow(one)
        plt.scatter([p[0] for p in suppressed_points_2], [p[1] for p in suppressed_points_2], s = 1,c='orange')
        plt.savefig("anms2.jpg")

        #descriptor finding
        descriptors_1 = find_descriptors(one_bw, suppressed_points_1)
        descriptors_2 = find_descriptors(two_bw, suppressed_points_2)

        #feature matching
        matched_features = match_features(descriptors_1, descriptors_2,th=0.5)

        matched_one = matched_features.keys()
        plt.figure()
        plt.imshow(one)
        plt.scatter([p[0] for p in matched_one], [p[1] for p in matched_one], s=10, c='orange')
        plt.savefig("matched1.jpg")

        matched_two = matched_features.values()
        plt.figure()
        plt.imshow(two)
        plt.scatter([p[0] for p in matched_two], [p[1] for p in matched_two], s=10, c='orange')
        plt.savefig("matched2.jpg")

        #ransac
        ransac_points = generate_ransac_points(matched_features,err_th=0.5,n=1000)
        p_one = list(ransac_points.keys())
        p_two = list(ransac_points.values())
        print("selected keypoints:")
        print("p_one: {} len:{}".format(p_one,len(p_one)))
        print("p_two: {} len:{}".format(p_two,len(p_two)))

        H = computeH(np.asarray(p_one),np.asarray(p_two))
        print("homography: {}".format(H))

        warpImage(one,two,H)
        exit(0)

    else:
        """
        MANUAL SELECTION OF KEYPOINTS
        """
        
        print('manual selection')
        #get points
        p_one, p_two = getPoints(one,two)

        #plot images with their keypoints
        plt.figure()
        plt.imshow(one)
        one_x,one_y = list(zip(*p_one))
        plt.scatter(one_x,one_y)
        plt.savefig("one_points.jpg")

        plt.figure()
        plt.imshow(two)
        two_x,two_y = list(zip(*p_two))
        plt.scatter(two_x,two_y)
        plt.savefig("two_points.jpg")

        H = computeH(np.asarray(p_one),np.asarray(p_two))
        print("homography: {}".format(H))

        warpImage(one,two,H)
        exit(0)

