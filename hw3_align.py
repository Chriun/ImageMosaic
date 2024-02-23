import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
"""
Sources/References: OpenCV Docs, Class Jupyter Notebooks
"""

"""
    get_images: Finds all images in a given directory and returns list containing their name and
                list containing the actual images read in grayscale
"""
def get_images(img_dir):
    original = os.getcwd()
    os.chdir(img_dir)
    img_name_list = os.listdir('./')
    img_name_list = [name for name in img_name_list if 'jpg' in name.lower()]
    img_name_list.sort()

    img_list = []
    for i in img_name_list:
        im = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print('Could not open', i)
            sys.exit(0)
        img_list.append(im)

    os.chdir(original)
    return img_name_list, img_list

"""
Step 1 Utilities
    key_desc_extractor: Utilizes SIFT alg to detect keypoints in an image and computes descriptors based on keypoints.
                        Returns both keypoints and descriptors.

    printKP: Outputs # of keypoints for each image
"""
def key_desc_extractor(img):
    sift = cv2.SIFT_create()
    kp = sift.detect(img, None)
    #sift.compute returns keypoints and descriptors
    #could have used detectAndCompute instead
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

    #if the two pass the ratio test, find corresponding point in keypoints
    for m,n in matches:
        if (m.distance/n.distance) < 0.8:
            passed.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[n.trainIdx].pt)

    return passed, pts1, pts2

def printMatch(im1, im2, kp1, kp2, matches):
    print("Step 2")
    print("\tMatches:", len(pts1))
    print("\tFraction in Image 1: " + str(len(pts1)) + "/" + str(len(kp1)) + " = " + str(len(pts1)/len(kp1)))
    print("\tFraction in Image 2: " + str(len(pts2)) + "/" + str(len(kp2)) + " = " + str(len(pts2)/len(kp2)))
    sideBySide = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.axis('off')
    plt.imshow(sideBySide)
    plt.show()
    # img = im1 + im2 + "Step2b.jpg"
    # cv2.imwrite(img, sideBySide)

"""
Step 4 Utilities
    findFundamentalMatrix: Finds the Fundamental Matrix via RANSAC and utilizes the mask of inliers
                           to find corresponding points in both images that lie in epipolar line.
                           Returns both image's inliers and their corresponding DMatch objects.

    printInliers: Outputs the number of inliers and the percentage over matches. 
                  Displays side-by-side image with lines drawn between matching inlier keypoints.
"""
def findFundamentalMatrix(pts1, pts2, matches):
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    matches = np.asarray(matches)

    #cv2 arguments take in pts array as np array
    #outputs F matrix, and mask of inliers (shape (# matches, 1))
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    mask = mask.ravel()
    pts1 = pts1[mask == 1]
    pts2 = pts2[mask == 1]
    matches = matches[mask == 1]
    matches = matches.tolist()

    return pts1, pts2, matches

def printInliers(im1, im2, pts1, pts2, kp1, kp2, inliers1, inliers2, matches):
    print("Step 4")
    print("\tInliers:", len(inliers1))
    print("\tPercentage of inliers over matches:", len(inliers1)/len(pts1))
    inlierMatch = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.axis('off')
    plt.imshow(inlierMatch)
    plt.show()
    # img = im1 + im2 + "Step4b.jpg"
    # cv2.imwrite(img, inlierMatch)

"""
Step 6 Utilities
    findFundamentalMatrix: Finds the Fundamental Matrix via RANSAC and utilizes the mask of inliers
                           to find corresponding points in both images that lie in epipolar line.
                           Returns both image's inliers and their corresponding DMatch objects.

    printInliers: Outputs the number of inliers and the percentage over matches. 
                  Displays side-by-side image with lines drawn between matching inlier keypoints.
"""
def findHomography():
    pass

"""Main"""
in_dir = sys.argv[1]
out_dir = sys.argv[2]
img_name_list, img_list = get_images(in_dir)

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
        printMatch(im1, im2, kp1, kp2, matches)
        """Step3 temp"""
        print("Step 3")
        if len(pts1)/len(dc1) < 0.1:
            print("\tToo small a percentage of matches, Next image")
            print()
            print("------------------------------------------------------")
            print()
            continue
        print("\tPassed threshold")
        """Step4, finding Fundamental Matrix via Ransac and finding inliers of epipolar line, printing inliers"""
        inliers1, inliers2, matches = findFundamentalMatrix(pts1, pts2, matches)
        printInliers(im1, im2, pts1, pts2, kp1, kp2, inliers1, inliers2, matches)
        """Step5 temp"""
        print("Step 5")
        if len(inliers1)/len(pts1) < 0.1:
            print("\tToo small a percentage of matches, Next image")
            print()
            print("------------------------------------------------------")
            print()
            continue
        print("\tPassed threshold")
        """Step6"""
        findHomography()
        print()
        print("------------------------------------------------------")
        print()

