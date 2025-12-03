import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from stitch import Stitcher
    
if __name__ == "__main__":
    # path1 = 'imgs/part1/stiching_1/CS1.jpg'
    # path2 = 'imgs/part1/stiching_1/CS2.jpg'
    # path1 = 'imgs/part1/stiching_2/UOB1.jpg'
    # path2 = 'imgs/part1/stiching_2/UOB2.jpg'
    # path1 = 'imgs/part1/stiching_3/Door2.jpeg'
    # path2 = 'imgs/part1/stiching_3/Door1.jpeg'
    path1 = 'imgs/part2/demo1.jpg'
    path2 = 'imgs/part2/demo2.jpg'
    img_left = cv2.imread(path1)
    img_right = cv2.imread(path2)
    img_left = cv2.resize(img_left, (1000, 750))
    img_right = cv2.resize(img_right, (1000, 750))

    # The stitch object to stitch the image
    blending_mode = "linearBlending" # three mode - noBlending、linearBlending、linearBlendingWithConstant
    stitcher = Stitcher()
    SIFT_thresh = [0.01, 0.04, 0.08, 0.12, 0.16]
    for value in SIFT_thresh:
        HomoMat, warp_img, inliers = stitcher.stitch([img_left, img_right], 
                                            detector="ORB",
                                            blending_mode=blending_mode,
                                            SIFT_thresh=value)
        print(f"Inliers: {inliers}")
    
    cv2.imwrite('results/stitch_img.png', warp_img)