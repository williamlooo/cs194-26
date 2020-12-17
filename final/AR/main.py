import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import os
import pickle
import skvideo.io
import skimage.io

def getPoints(im,num=8):
    print('Please select {} points in each image for alignment.'.format(num))
    points = []
    plt.imshow(im)
    points = np.asarray(plt.ginput(num,timeout=0))
    plt.close()
    return points

def get3DPoints(im, first_img_points):
    points_3D = []
    c=0
    for point in first_img_points:
        plt.imshow(im)
        plt.scatter(point[0], point[1])
        plt.show(block=False)
        x = input("enter X: ")
        y = input("enter Y: ")
        z = input("enter Z: ")
        print("entered ({}, {}, {}) for point {}.".format(x,y,z,c))
        points_3D.append(np.array([int(x), int(y), int(z)]))
        plt.close()
        c+=1
    return points_3D

def draw(img, imgpts):
    """
    taken from https://docs.opencv.org/3.4/d7/d53/tutorial_py_pose.html
    """
    imgpts = imgpts[:, :2].astype(np.int)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0), -3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

if __name__ == "__main__":
    """
    PART 1
    """
    video = skvideo.io.vread("input.mp4")
    first_frame = video[0]

    # 1. first frame get points
    NUM_POINTS = 50

    if os.path.exists("points.p"):
        print("[INFO] selected camera points exist, loading..")
        first_img_points = pickle.load(open("points.p","rb"))
    else:
        first_img_points = getPoints(first_frame,NUM_POINTS)
        pickle.dump(first_img_points,open("points.p","wb"))
    assert len(first_img_points) == NUM_POINTS

    """
    plot selected img points to selected_pts.png
    """
    plt.figure()
    first_img_points_x = [i[0] for i in first_img_points]
    first_img_points_y = [i[1] for i in first_img_points]
    plt.imshow(first_frame)
    plt.scatter(first_img_points_x,first_img_points_y)
    plt.savefig("selected_pts.png")
    plt.close()

    # 2. get corresponding 3d coordinates for these points 
    if os.path.exists("points_3D.p"):
        print("[INFO] 3D points exist, loading..")
        points_3D = pickle.load(open("points_3D.p","rb"))
    else:
        points_3D = get3DPoints(first_frame, first_img_points)
        pickle.dump(points_3D,open("points_3D.p","wb"))
    assert len(points_3D) == NUM_POINTS

    # 3 . create bounding boxes for each image point
    bboxes = []
    BBOX_SIZE = 20
    for point in first_img_points:
        bboxes.append(np.array([int(point[0]-(BBOX_SIZE/2)), int(point[1]-(BBOX_SIZE/2)),BBOX_SIZE, BBOX_SIZE]))

    # 4. set up tracker for each point 
    trackers = []
    for bbox in bboxes:
        #new_tracker = cv2.TrackerMedianFlow_create()
        new_tracker = cv2.TrackerCSRT_create()
        new_tracker.init(first_frame, tuple(bbox))
        trackers.append(new_tracker)

    # 5. for each frame in video, get tracker result
    if os.path.exists("tracked_points_all_frames.p") and os.path.exists("points_3D_all_frames.p"):
        print("[INFO] selected tracked and 3D points for all frames exist, loading..")
        tracked_points_all_frames = pickle.load(open("tracked_points_all_frames.p","rb"))
        points_3D_all_frames = pickle.load(open("points_3D_all_frames.p","rb"))

    else:
        tracked_points_all_frames = [[] for _ in range(len(video))]
        points_3D_all_frames = [[] for _ in range(len(video))]
        if not os.path.exists("keypoints"):
            os.makedirs("keypoints")
        for frame_num in range(len(video)):
            frame = video[frame_num]
            point_id=0
            for tracker in trackers:
                stat,bbox = tracker.update(frame)
                
                if stat and bbox[0] > 10 and bbox[1] > 10:
                    #filter by bbox position (if out of bound, do not add)    
                    if bbox[2] < 30 and bbox[3] < 30:
                        #filter by bbox size (if too big, toss result)
                        #point should only be used if it is detected and reasonably accurate
                        tracked_points_all_frames[frame_num].append(np.array([int(bbox[0] + (bbox[2]/2)),
                                                                    int(bbox[1] + (bbox[3]/2))]))
                        points_3D_all_frames[frame_num].append(points_3D[point_id])
                point_id+=1

                tl = (int(bbox[0]), int(bbox[1]))
                br = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, tl, br, (255, 0, 0), 3)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite("keypoints/{}.jpg".format(frame_num), frame)

        pickle.dump(tracked_points_all_frames,open("tracked_points_all_frames.p","wb"))
        pickle.dump(points_3D_all_frames,open("points_3D_all_frames.p","wb"))

    # at this point make sure the points are following the video movement 

    """
    PART 2
    """
    # 6. compute camera calibrated matrix
    cube_points = np.float32([[1,1,2,1], [1,2,2,1], [2,2,2,1], [2,1,2,1],
                        [1,1,3,1],[1,2,3,1],[2,2,3,1],[2,1,3,1]]).T
    print("[INFO] saving video of projected cube into directory 'output_cube/'..")
    if not os.path.exists("output_cube"):
        os.makedirs("output_cube")
    for frame_num in range(len(video)):
        frame = video[frame_num]
        frame_points = np.array(tracked_points_all_frames[frame_num])
        frame_points_3D = np.array(points_3D_all_frames[frame_num])

        assert(len(frame_points) == len(frame_points_3D))
        frame_points_normalized = np.hstack([frame_points,    np.ones((frame_points.shape[0], 1))])
        frame_points_3D_normalized = np.hstack([frame_points_3D, np.ones((frame_points_3D.shape[0], 1))])
        projection_matrix = lstsq(frame_points_3D_normalized, frame_points_normalized, rcond=None)[0]

        # 7. use projection matrix to draw the box
        transformed_cube_points = projection_matrix.T.dot(cube_points)
        transformed_cube_points /= transformed_cube_points[2]
        transformed_cube_points = transformed_cube_points.T
        res = draw(frame, transformed_cube_points)
        # at this point the box should be tracked. 
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        cv2.imwrite("output_cube/{}.jpg".format(frame_num),res)

    print("done.")
