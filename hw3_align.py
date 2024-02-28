import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
"""
Sources/References: OpenCV Docs,
                    In particular:
                        SIFT,
                        DrawMatches,
                        Flann and BF Matchers,
                        findFundamentalMat,
                        findHomography,
                        warpPerspective,
                        addWeighted,
                        Epipolar lines tutorial
                    Class Jupyter Notebooks,
                    Discussed with classmates
"""

"""
    get_images: Finds all images in a given directory and returns list containing their name and
                list containing the actual images read in grayscale.
                Also checks if output directory exists and if not, creates it.
"""
def get_images(img_dir, out_dir):
    original = os.getcwd()
    os.chdir(img_dir)
    img_name_list = os.listdir('./')
    img_name_list = [name for name in img_name_list if 'jpg' in name.lower()]
    img_name_list.sort()

    img_list = []
    colored_img_list = []
    for i in img_name_list:
        im = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        color_im = cv2.imread(i)
        if im is None:
            print('Could not open', i)
            sys.exit(0)
        img_list.append(im)
        colored_img_list.append(color_im)

    os.chdir(original)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return img_name_list, img_list, colored_img_list

"""
Step 1 Utilities
    key_desc_extractor: Utilizes SIFT alg to detect keypoints in an image and computes descriptors based on keypoints.
                        Returns both keypoints and descriptors.

    printKP: Outputs # of keypoints for each image
"""
def key_desc_extractor(img):
    sift = cv2.SIFT_create()
    kp = sift.detect(img, None)
    # sift.compute returns keypoints and descriptors
    # could have used detectAndCompute instead
    kp, desc = sift.compute(img, kp)
    return kp, desc

def printKP(im1, im2, kp1, kp2):
    print("Images: " + im1 + ", " + im2)
    print()
    print("Step 1")
    print("\tKeypoints in " + im1 + ": " + str(len(kp1)))
    print("\tKeypoints in " + im2 + ": " + str(len(kp2)))
    print()

"""
Step 2 Utilities
    match: Utilizes FLANN alg to figure out 2 best DMatch objects and use their distance attribute to 
           conduct ratio test. Passed matches are returned as well as corresponding points in both images.
    
    printMatch: Outputs # of matches and the percentages of matches in each image over # of keypoints.
                Displays side-by-side image of both images with lines drawn between matching keypoints.
"""
def match(kp1, kp2, dc1, dc2):
    # bf = cv2.BFMatcher()
    # #returns tuple of top 2 best DMatch matches (distance to closest descriptor, second closest)
    # matches = bf.knnMatch(dc1, dc2, k = 2)

    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #returns tuple of top 2 best DMatch object matches (distance to closest descriptor, second closest)
    matches = flann.knnMatch(dc1,dc2,k=2)

    passed = []
    pts1 = []
    pts2 = []

    #if the two pass the ratio test, find corresponding point in keypoints (using best distance)
    for m,n in matches:
        if (m.distance/n.distance) < 0.8:
            passed.append(m)    
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    return passed, pts1, pts2

def printMatch(im1, im2, im1Name, im2Name, kp1, kp2, matches, out_dir):
    print("Step 2")
    print("\tMatches:", len(pts1))
    print("\tFraction in Image 1: " + str(len(pts1)) + "/" + str(len(kp1)) + " = " + str(len(pts1)/len(kp1)))
    print("\tFraction in Image 2: " + str(len(pts2)) + "/" + str(len(kp2)) + " = " + str(len(pts2)/len(kp2)))
    sideBySide = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    #write img to out_dir
    im1Name = im1Name.split('.')
    img = im1Name[0] + im2Name.split('.')[0] + "Step2b." + im1Name[1]
    original = os.getcwd()
    os.chdir(out_dir)
    cv2.imwrite(img, sideBySide)
    os.chdir(original)

"""
Step 4 Utilities
    findFundamentalMatrix: Finds the Fundamental Matrix via RANSAC and utilizes the mask of inliers
                           to find corresponding points in both images that lie in epipolar line.
                           Returns both image's inliers and their corresponding DMatch objects.

    printF_Inliers: Outputs the number of inliers and the percentage over matches from Fundamental.
                    Displays side-by-side image with lines drawn between matching F inlier keypoints.

    drawLines: Draws the lines that intersect matching points in each image
    drawImage: Creates the side by side image of epipolar lines
"""
def findFundamentalMatrix(pts1, pts2, matches):
    matches = np.asarray(matches)

    #cv2 arguments take in pts array as np array
    #outputs F matrix, and mask of inliers (shape (# matches, 1))
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    #flattens mask to match points array shape
    mask = mask.ravel()

    #finds points and matches that are inliers in Fundamental matrix
    inliers1 = pts1[mask == 1]
    inliers2 = pts2[mask == 1]
    matches = matches[mask == 1]
    matches = matches.tolist()

    return F, inliers1, inliers2, matches

def printF_Inliers(im1, im2, im1Name, im2Name, pts1, kp1, kp2, inliers1, matches, out_dir):
    print("Step 4")
    print("\tInliers:", len(inliers1))
    print("\tPercentage of F inliers over matches:", len(inliers1)/len(pts1))
    F_inlierMatch = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
    im1Name = im1Name.split('.')
    img = im1Name[0] + im2Name.split('.')[0] + "Step4b." + im1Name[1]
    original = os.getcwd()
    os.chdir(out_dir)
    cv2.imwrite(img, F_inlierMatch)
    os.chdir(original)

def drawLines(im1, lines, pts):
    _,col = im1.shape
    im1 = cv2.cvtColor(im1,cv2.COLOR_GRAY2BGR)
    #calculates the lines that intersect each matching keypoint
    for (a,b,c),pt in zip(lines,pts):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -c/b ])
        x1,y1 = map(int, [col, -(c+a*col)/b ])
        im1 = cv2.line(im1, (x0,y0), (x1,y1), color,1)
        im1 = cv2.circle(im1,tuple(map(np.int32, pt)),5,color,-1)
    return im1

def drawImages(im1, im2, im1Name, im2Name):
    #concatenate images to create side by side
    sideBySide = np.concatenate((im1, im2), axis = 1)
    im1Name = im1Name.split('.')
    img = im1Name[0] + im2Name.split('.')[0] + "Step4c." + im1Name[1]
    original = os.getcwd()
    os.chdir(out_dir)
    cv2.imwrite(img, sideBySide)
    os.chdir(original)
    

"""
Step 6 Utilities
    findHomography: Finds the Homography Matrix via RANSAC as well as inliers from Fundamental Matrix
                    and utilizes the mask of inliers to find corresponding points that map correctly 
                    onto the other image.
                    Returns both image's inliers and their corresponding DMatch objects.

    printH_Inliers: Outputs the number of H inliers and the percentage over inlier matches from F.
                    Displays side-by-side image with lines drawn between matching H inlier keypoints.
"""
def findHomography(F_inliers1, F_inliers2, matches):
    matches = np.asarray(matches)

    H, mask = cv2.findHomography(F_inliers1, F_inliers2, cv2.FM_RANSAC)
    
    #flatten mask to correspond with points shape
    mask = mask.ravel()
    
    #finds the points and matches that are inliers to Homography
    H_inliers1 = F_inliers1[mask == 1]
    H_inliers2 = F_inliers2[mask == 1]
    matches = matches[mask == 1]
    matches = matches.tolist()

    return H, H_inliers1, H_inliers2, matches

def printH_Inliers(im1, im2, im1Name, im2Name, kp1, kp2, F_inliers1, H_inliers1, matches, out_dir):
    print("Step 6")
    print("\tInliers:", len(H_inliers1))
    print("\tPercentage of H inliers over F inliers:", len(H_inliers1)/len(F_inliers1))
    H_inlierMatch = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    im1Name = im1Name.split('.')
    img = im1Name[0] + im2Name.split('.')[0] + "Step6b." + im1Name[1]
    original = os.getcwd()
    os.chdir(out_dir)
    cv2.imwrite(img, H_inlierMatch)
    os.chdir(original)

"""
Step8 Utilities
    createMosaic: Creates bounding box and constructs a new homography matrix to reflect the
                  translation occurred of image 2 in that bounding box. 
                  Warps image 1 into a new bounding box field and combines the two bounding boxes
                  to finally create the mosaic.

    createBoundingBox: Maps corners of image 1 to image 2 coords to create the bounding box.
"""
def createMosaic(H, im1, im2, im1Name, im2Name):
    R, shift = createBoundingBox(H, im1, im2)

    #create a simple translation homography matrix and multiplies it by our H to reflect translated coordinates (im2 coords)
    H_shift = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    new_H = H_shift @ H
    
    #warps image 1 into a new bounding box image (separate from the bounding box with image 2)
    mosaic = cv2.warpPerspective(im1, new_H, (R.shape[1], R.shape[0])).astype(np.int32)

    #multiplies intensities in regions where the opposite image does not reside in (to cancel out addWeighted)
    R[mosaic == 0] = R[np.where(mosaic == 0)]*2
    mosaic[R == 0] = mosaic[np.where(R == 0)]*2

    #addWeighted reduces each images intensity by half and combines them
    mosaic = cv2.addWeighted(mosaic, 0.5, R, 0.5, 0).astype(np.int32)

    #writes img into out_directory
    im1Name = im1Name.split('.')
    img = im1Name[0] + '_' + im2Name.split('.')[0] + '.' + im1Name[1]

    original = os.getcwd()
    os.chdir(out_dir)
    cv2.imwrite(img, mosaic)
    os.chdir(original)

    return img

def createBoundingBox(H, im1, im2):
    x, y, _ = im1.shape
    #mapping im1 corners to im2
    corners = np.array([[0,0,1],[x,0,1],[0,y,1],[x,y,1]])
    corners = H @ corners.T

    #divide by homogenous coord, z
    corners = corners / corners[2, :]
    corners = corners.T

    #finding the min and max bounds of mosaic (in the end, its the min x and y of each corner)
    minBound = np.min(corners, axis = 0)[:2].astype(np.int32)
    
    x_min = abs(min(minBound[0], 0))
    y_min = abs(min(minBound[1], 0))

    maxBound = np.max(corners, axis = 0)[:2].astype(np.int32)  
    
    #need to swap x and y to account for opencv !! 
    #adding min with max bound to get tight bounding box
    x_max = abs(max(maxBound[0], im2.shape[1])) + x_min
    y_max = abs(max(maxBound[1], im2.shape[0])) + y_min

    #creates bounding box and places image 2 into its respective position
    R = np.zeros((int(y_max), int(x_max), 3)).astype(np.int32)
    R[y_min:y_min + im2.shape[0], x_min:x_min + im2.shape[1]] = im2

    return R, (x_min, y_min)

"""Main"""
in_dir = sys.argv[1]
out_dir = sys.argv[2]
img_name_list, img_list, colored_img_list = get_images(in_dir, out_dir)

print("------------------------------------------------------")
"""looping through images, comparing Image i and j where 1<=i<j<=N"""
for i in range(len(img_list)-1):
    for j in range(i+1, len(img_list)):
        """Step 1, extracting kp and desc, printing kp"""
        im1 = img_list[i]
        im2 = img_list[j]
        kp1, dc1 = key_desc_extractor(im1)
        kp2, dc2 = key_desc_extractor(im2)
        printKP(img_name_list[i], img_name_list[j], kp1, kp2)

        """Step2, matching kp and desc using FLANN, filtering through ratio test, printing matches"""
        matches, pts1, pts2 = match(kp1, kp2, dc1, dc2)
        printMatch(im1, im2, img_name_list[i], img_name_list[j], kp1, kp2, matches, out_dir)

        """Step3 temp"""
        print("Step 3")
        if len(pts1)/len(dc1) < 0.05:
            print("\tToo small a percentage of matches, Different Scenes")
            print()
            print("------------------------------------------------------")
            print()
            continue
        print("\tPassed threshold")

        """Step4, finding Fundamental Matrix via Ransac and finding inliers of epipolar line, printing inliers, draws epipolar lines"""
        F, F_inliers1, F_inliers2, Fmatches = findFundamentalMatrix(pts1, pts2, matches)
        printF_Inliers(im1, im2, img_name_list[i], img_name_list[j], pts1, kp1, kp2, F_inliers1, Fmatches, out_dir)
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        lines1 = lines1.reshape(-1,3)
        epi1 = drawLines(im1, lines1, pts1)

        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        epi2 = drawLines(im2, lines2, pts2)

        drawImages(epi1, epi2, img_name_list[i], img_name_list[j])

        """Step5 temp"""
        print("Step 5")
        if len(F_inliers1)/len(pts1) < 0.8:
            print("\tToo small a percentage of inliers, Different Scenes")
            print()
            print("------------------------------------------------------")
            print()
            continue
        print("\tPassed threshold")

        """Step6"""
        H, H_inliers1, H_inliers2, Hmatches = findHomography(F_inliers1, F_inliers2, Fmatches)
        printH_Inliers(im1, im2, img_name_list[i], img_name_list[j], kp1, kp2, F_inliers1, H_inliers1, Hmatches, out_dir)

        """Step7 temp"""
        print("Step 7")
        if len(H_inliers1)/len(F_inliers1) < 0.65:   
            print("\tNo. Images cannot be accurately aligned. There are less than 0.65 percent of inliers.")
            print()
            print("------------------------------------------------------")
            print()
            continue
        print("\tYes. Images can be accurately aligned. Over 0.65 percent inliers.")

        """Step8"""
        print("Step 8")
        im1 = colored_img_list[i]
        im2 = colored_img_list[j]
        img = createMosaic(H, im1, im2, img_name_list[i], img_name_list[j])
        print("\tCreated mosaic:", img)

        print()
        print("------------------------------------------------------")
        print()