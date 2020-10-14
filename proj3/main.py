import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import skimage.io as skio
from scipy.interpolate import interp2d
from skimage.draw import polygon
import pickle
import os

def get_points(im1, im2):
    print('Please select 29 points in each image for alignment.')
    points_1 = []
    points_2 = []
    plt.imshow(im1)
    points_1 = plt.ginput(29,timeout=0) #58 for danish IMM
    plt.close()
    plt.imshow(im2)
    points_2 = plt.ginput(29,timeout=0)
    print("im1")
    print(points_1)
    print("im2")
    print(points_2)
    plt.close()
    return points_1, points_2

def computeAffine(tri1_pts,tri2_pts):
    homo_one = np.hstack([tri1_pts, [[1], [1], [1]]])
    homo_two = np.hstack([tri2_pts, [[1], [1], [1]]])
    #print(np.linalg.solve(homo_one,homo_two).T)
    #print("\n")
    #print("linalg solve transpose")
    #print(np.linalg.solve(homo_one.T,homo_two.T))
    return np.linalg.solve(homo_two,homo_one)
    #print((np.linalg.inv(homo_one)@homo_two).T)

def applyWarp(img, start_keypts, end_keypts, tri):
    res = np.zeros(img.shape)
    height=img.shape[0]
    width=img.shape[1]

    interpolate_red = interp2d(range(width), range(height), img[:,:,0],kind='linear')
    interpolate_green = interp2d(range(width), range(height), img[:,:,1],kind='linear')
    interpolate_blue = interp2d(range(width), range(height), img[:,:,2],kind='linear')

    start_tri_points = start_keypts[tri.simplices]
    end_tri_points = end_keypts[tri.simplices]

    for start_pt, end_pt in zip(start_tri_points, end_tri_points):
        canvas = np.zeros(img.shape)
        transform = computeAffine(start_pt, end_pt).T

        row = [i[0] for i in end_pt]
        col = [i[1] for i in end_pt]

        rr, cc = polygon(row, col)
        canvas[cc, rr] = 1

        selected = np.where(canvas)
        selected_row_indices = selected[0]
        selected_col_indices = selected[1]

        homo_selected_indices = np.vstack([selected_col_indices, selected_row_indices,np.ones(selected_row_indices.shape)]).astype(int)

        original_mapping = (transform @ homo_selected_indices).astype(int) #transform and round to nearest int
        """
        #took too long to interpolate so casted to int instead.
        red_channel = interpolate_red(original_mapping[0], original_mapping[1])
        print(red_channel)
        green_channel = interpolate_green(original_mapping[0], original_mapping[1])
        print(green_channel)
        blue_channel = interpolate_blue(original_mapping[0], original_mapping[1])
        print(blue_channel)
        #res[warped_points[1], warped_points[0], :] = [red_channel, green_channel, blue_channel]
        """
        res[homo_selected_indices[1].astype(int), homo_selected_indices[0].astype(int),:] = img[original_mapping[1].astype(int), original_mapping[0].astype(int), :]
    return res

#morphed_im = morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac);
def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    warped_pts = (warp_frac)* im1_pts+((1-warp_frac)*im2_pts)
    tri = Delaunay(warped_pts)

    one = applyWarp(im1, im1_pts, warped_pts, tri)
    two = applyWarp(im2, im2_pts, warped_pts, tri)
    res = (dissolve_frac*one)+((1-dissolve_frac)*two)
    return res

def morph_one_image(im1, im1_pts, im2_pts, warp_frac):
    warped_pts = (warp_frac)* im1_pts+((1-warp_frac)*im2_pts)
    tri = Delaunay(warped_pts)

    one = applyWarp(im1, im1_pts, warped_pts, tri)
    #two = applyWarp(im2, im2_pts, warped_pts, tri)
    #res = (dissolve_frac*one)+((1-dissolve_frac)*two)
    return one

def make_movie(im1, im2, im1_pts, im2_pts, filename):
    frames = np.arange(0,1,0.0125)
    avg = (0.5*im1_pts) + (0.5*im2_pts)
    tri = Delaunay(avg)

    count=0
    for f in list(frames):
        out_frame = morph(im1, im2, im1_pts, im2_pts, tri, f, f)
        output_filename = "{}-{}.jpg".format(filename,str(count))
        skio.imsave(output_filename, out_frame)
        count+=1

def getMeanFace():
    #640x480
    corners = [(0,0),(640,0),(0,480),(640,480)]
    width = 640
    height=480
    directory="imm_face_db"
    dataset_keypoints = []
    filename_img_dict = {}
    avg = np.zeros((height,width,3))
    for filename in os.listdir(directory):
        #if filename.endswith(".jpg"): 
        #end = filename.split('-')[1]
        if filename.endswith("1m.jpg") or filename.endswith("1f.jpg"): #full frontal face, neutral expression, diffuse light
            filename_without_ext = filename.split(".")[0]    
            with open("{}/{}.asf".format(directory,filename_without_ext)) as fp:
                points = []
                for cnt, line in enumerate(fp):
                    tokenized = line.split("\t")
                    if len(tokenized) == 7:
                        #print(filename)
                        x_pos = (float(tokenized[2])*width)
                        y_pos = (float(tokenized[3])*height)
                        #print((x_pos,y_pos))
                        points.append((x_pos,y_pos))

                if len(points) == 58:
                    points.extend(corners) #add corner points
                    filename_img_dict[filename] = points
                    dataset_keypoints.append(points)

    mean_points = np.mean(dataset_keypoints,0)
    tri = Delaunay(mean_points)
    print(mean_points) 

    for filename in filename_img_dict.keys():
        img = skio.imread("{}/{}".format(directory,filename))
        start_keypts = filename_img_dict[filename]
        print(filename)
        res = applyWarp(img, np.asarray(start_keypts), mean_points, Delaunay(mean_points))
        #res = morph(img, img, np.asarray(start_keypts), mean_points, Delaunay(mean_points), 0.5, 0);
        skio.imsave("averages/{}".format(filename),res)
        avg = (avg+res)

    avg = avg/len(filename_img_dict.keys())
    skio.imsave('avg.jpg',avg)
    plt.triplot(mean_points[:,0], mean_points[:,1], tri.simplices)
    plt.imshow(avg)
    plt.savefig('out.png')
    return mean_points

if __name__ == '__main__':
    george = skio.imread("george_small.jpg")
    will = skio.imread("will.jpg")
    will_danish = skio.imread("will_danish.jpg") #same as will but different size
    p1 = [(4.233766233766062, 10.457792207792068), (593.7224025974026, 10.457792207792068), (145.26461038961025, 142.35714285714278), (288.3246753246752, 99.7435064935064), (450.6623376623375, 145.40097402597394), (117.87012987012969, 220.48214285714278), (478.0568181818181, 224.54058441558425), (101.63636363636346, 327.01623376623365), (144.24999999999983, 333.103896103896), (239.6233766233765, 333.103896103896), (338.0405844155843, 333.103896103896), (448.63311688311677, 342.23538961038946), (509.5097402597401, 339.1915584415583), (106.70941558441541, 415.28733766233756), (238.60876623376606, 445.7256493506493), (295.42694805194793, 427.46266233766227), (345.14285714285705, 460.9448051948051), (496.31980519480504, 421.3749999999999), (109.75324675324657, 499.49999999999994), (208.17045454545445, 546.1720779220778), (288.3246753246752, 524.8652597402597), (288.3246753246752, 568.4935064935064), (377.6103896103895, 542.1136363636363), (488.202922077922, 538.0551948051948), (149.32305194805184, 612.1217532467532), (290.3538961038959, 695.3198051948052), (433.41396103896096, 630.3847402597402), (1.1899350649349003, 748.0795454545454), (598.7954545454545, 745.0357142857142)]
    p2 = [(2.2045454545452685, 6.399350649350481), (597.7808441558441, 7.413961038960906), (144.24999999999983, 172.7954545454544), (309.6314935064933, 110.90422077922062), (466.89610389610374, 181.926948051948), (110.767857142857, 251.93506493506482), (506.46590909090895, 255.9935064935064), (121.92857142857127, 344.2646103896103), (166.57142857142838, 342.23538961038946), (232.52110389610374, 347.3084415584415), (353.2597402597401, 347.3084415584415), (457.76461038961025, 347.3084415584415), (522.6996753246751, 347.3084415584415), (107.72402597402584, 428.47727272727263), (240.6379870129869, 474.13474025974017), (297.45616883116867, 446.74025974025966), (350.21590909090895, 483.2662337662337), (510.52435064935054, 436.59415584415575), (114.82629870129853, 537.0405844155844), (206.1412337662336, 565.4496753246752), (288.3246753246752, 545.1574675324675), (288.3246753246752, 610.0925324675325), (385.7272727272726, 576.6103896103896), (487.18831168831156, 553.2743506493506), (175.70292207792187, 644.5892857142857), (297.45616883116867, 720.6850649350649), (426.3116883116882, 647.6331168831168), (4.233766233766062, 744.0211038961038), (592.7077922077922, 746.0503246753246)]

    assert(len(p1)==len(p2)) #same number of points (29)
    
    #POINT SELECTION
    print("POINT SELECT")
    #points_1, points_2 = get_points(george,will)

    #MIDWAY FACE
    print("MIDWAY FACE")
    points_1=np.asarray(p1)
    points_2=np.asarray(p2)

    ALPHA=0.5
    mid_pts = np.around(((1-ALPHA)*points_1)+((ALPHA)*points_2))

    tri = Delaunay(mid_pts)
    
    warp2 = applyWarp(george,points_1, mid_pts,tri)
    warp1 = applyWarp(will,points_2, mid_pts,tri)
    mid = (0.5*warp1)+(0.5*warp2)
    skio.imsave("mid.jpg",mid)

    #MORPHING 
    print("MORPHING")
    morphed_im = morph(george, will, points_1, points_2, tri, 0.5, 0.5);
    skio.imsave("morph.jpg",morphed_im)
    #make_movie(george, will, points_1, points_2, "video/george-will")

    #MEAN FACE
    print("MEAN FACE")
    danish_will_points = [(202.6168831168831, 359.7597402597402), (216.90259740259742, 384.43506493506493), (228.5909090909091, 398.72077922077915), (240.27922077922076, 413.0064935064935), (251.9675324675325, 428.59090909090907), (271.4480519480519, 445.47402597402595), (301.31818181818187, 454.564935064935), (329.8896103896104, 454.564935064935), (355.8636363636364, 441.57792207792204), (370.1493506493507, 425.99350649350646), (390.92857142857144, 407.8116883116883), (409.1103896103896, 388.3311688311688), (423.3961038961039, 366.2532467532467), (410.4090909090909, 233.78571428571422), (393.52597402597405, 223.39610389610385), (375.3441558441558, 211.70779220779212), (358.461038961039, 218.20129870129864), (342.87662337662334, 235.08441558441552), (361.0584415584416, 237.68181818181813), (377.9415584415584, 237.68181818181813), (390.92857142857144, 237.68181818181813), (219.49999999999997, 231.18831168831161), (227.2922077922078, 225.9935064935064), (251.9675324675325, 218.20129870129864), (267.5519480519481, 223.39610389610385), (275.3441558441558, 232.48701298701292), (259.7597402597403, 235.08441558441552), (244.17532467532467, 235.08441558441552), (233.7857142857143, 233.78571428571422), (336.38311688311694, 185.73376623376618), (351.96753246753246, 177.94155844155836), (381.8376623376623, 180.53896103896096), (396.12337662337666, 185.73376623376618), (415.6038961038961, 202.61688311688306), (283.1363636363636, 190.92857142857133), (259.7597402597403, 176.64285714285705), (231.1883116883117, 176.64285714285705), (215.60389610389612, 185.73376623376618), (207.8116883116883, 202.61688311688306), (262.3571428571429, 367.551948051948), (294.82467532467535, 353.2662337662337), (307.8116883116883, 362.3571428571428), (322.0974025974026, 354.564935064935), (358.461038961039, 368.8506493506493), (327.2922077922078, 385.73376623376623), (306.51298701298697, 389.6298701298701), (285.73376623376623, 387.0324675324675), (293.52597402597405, 248.0714285714285), (292.22727272727275, 277.94155844155836), (279.2402597402597, 293.52597402597394), (274.0454545454545, 318.20129870129864), (294.82467532467535, 323.39610389610385), (309.1103896103896, 328.590909090909), (325.9935064935065, 325.9935064935064), (344.17532467532465, 319.49999999999994), (344.17532467532465, 292.22727272727263), (320.7987012987013, 276.6428571428571), (318.2012987012987, 249.3701298701298)]
    danish_will_points.extend([(0,0),(640,0),(0,480),(640,480)]) #add corners

    #p1,p2 = get_points(will_danish,gary)
    danish_mean_points = getMeanFace()
    #pickle.dump(danish_mean_points,open( "danish_mean.pkl", "wb" ))
    #danish_mean_points = pickle.load(open("danish_mean.pkl","rb"))
    #make_movie(will_danish, will_danish, np.asarray(danish_will_points), danish_mean_points, "danish/will-danish")
    danish_will = applyWarp(will_danish, np.asarray(danish_will_points), danish_mean_points, Delaunay(danish_mean_points))
    skio.imsave("will_as_a_danish.jpg",danish_will)

    #danish mean morph to will
    danish_average = skio.imread("avg.jpg")
    danish_as_will = applyWarp(danish_average, danish_mean_points, np.asarray(danish_will_points), Delaunay(np.asarray(danish_will_points)))    
    skio.imsave("danish_as_will.jpg",danish_as_will)

    #will morph to danish meme
    danish_will = applyWarp(will_danish, np.asarray(danish_will_points), danish_mean_points, Delaunay(danish_mean_points))
    skio.imsave("will_as_a_danish.jpg",danish_will)

    #caricature
    cari = morph_one_image(will_danish, np.asarray(danish_will_points), danish_mean_points, -0.7)
    skio.imsave("will_caricature.jpg",cari)

    #bells and whistle
    print("CLASS MORPH")
    #gary = skio.imread("gary.jpg")
    #will = skio.imread("will_resized.jpg")
    #p1_will=[(0.7462121212120394, 4.011634199134164), (545.2537878787879, 4.011634199134164), (162.54274891774884, 101.24512987012974), (275.3336038961038, 65.4632034632034), (390.4580627705627, 111.35741341991331), (132.2058982683982, 153.36228354978346), (423.90638528138527, 161.9188311688311), (126.76082251082244, 249.04004329004323), (158.653409090909, 251.37364718614714), (221.66071428571422, 254.48511904761898), (315.7827380952381, 256.8187229437229), (395.90313852813847, 256.8187229437229), (439.4637445887446, 252.92938311688306), (120.5378787878787, 345.49567099567093), (220.1049783549783, 358.71942640692635), (269.8885281385281, 337.7169913419913), (313.44913419913416, 364.1645021645021), (428.573593073593, 340.8284632034632), (133.7616341991341, 433.394751082251), (200.65827922077915, 432.61688311688306), (265.9991883116883, 419.39312770562765), (259.77624458874453, 457.50865800865796), (322.7835497835497, 435.7283549783549), (417.6834415584415, 427.94967532467524), (184.3230519480519, 497.9577922077922), (259.77624458874453, 543.8520021645021), (364.78841991341983, 492.51271645021643), (0.7462121212120394, 573.4109848484848), (543.6980519480519, 573.4109848484848)]
    #p2_gary=[(-0.03165584415592093, 0.12229437229427731), (546.0316558441558, 3.2337662337661186), (138.42884199134193, 107.46807359307354), (252.77543290043286, 102.02299783549779), (370.2334956709956, 105.91233766233756), (110.42559523809518, 158.02949134199127), (407.5711580086579, 151.0286796536796), (99.53544372294363, 273.15395021645014), (153.98620129870125, 272.3760822510822), (218.54924242424238, 272.3760822510822), (296.33603896103887, 273.15395021645014), (359.3433441558441, 266.9310064935064), (415.34983766233756, 266.9310064935064), (113.53706709956703, 350.16287878787875), (219.3271103896103, 352.4964826839826), (260.5541125541125, 328.3825757575757), (301.0032467532467, 354.0522186147186), (413.7941017316017, 326.0489718614718), (132.98376623376618, 414.72591991341983), (209.2148268398268, 413.9480519480519), (259.77624458874453, 396.05708874458867), (259.77624458874453, 430.28327922077915), (311.8933982683982, 409.2808441558441), (395.12527056277054, 406.16937229437224), (176.54437229437224, 475.39962121212113), (260.5541125541125, 507.29220779220776), (362.4548160173159, 466.84307359307354), (2.30194805194796, 574.1888528138528), (544.4759199134198, 573.4109848484848)]
    #make_movie(will, gary, p1_will, p2_gary, "class_morph/will_gary")

    #APPENDIX
    #debug printing for triangles
    #plt.triplot(points_2[:,0], points_2[:,1], tri.simplices)
    #plt.plot(points_2[:,0], points_2[:,1], 'o')
    #plt.imshow(will)

    print("done.")
