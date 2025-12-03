import cv2
import numpy as np
import matplotlib.pyplot as plt


class image_stitch:
    def __init__(self, img_path1, img_path2, SIFT=None, ORB=None, resize=None):
        self.img1 = cv2.imread(img_path1)
        self.img2 = cv2.imread(img_path2)
        if resize:
            self.img1 = cv2.resize(self.img1, (1000, 750))
            self.img2 = cv2.resize(self.img2, (1000, 750))
        self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB)
            
        self.SIFT = SIFT
        self.ORB = ORB
        if SIFT:
            # Initialize SIFT detector
            self.sift = cv2.SIFT_create()    
        if ORB:
            # Initialize ORB
            self.orb = cv2.ORB_create()

    def feature_detect(self):
        # Detect keypoints and descriptors
        # If mask, we only detect the region which is white (value = 255)
        if self.SIFT:
            kp1, des1 = self.sift.detectAndCompute(image=self.img1, mask=None)
            kp2, des2 = self.sift.detectAndCompute(image=self.img2, mask=None)
        if self.ORB:
            kp1, des1 = self.orb.detectAndCompute(self.img1, mask=None)
            kp2, des2 = self.orb.detectAndCompute(self.img2, mask=None)
        return (kp1, des1), (kp2, des2)
    
    def match(self, features_1, features_2):
        # Use FLANN to conduct KNN match（k=2, Best match and second match）
        # Using FLANN is for speed, if use brutal force it would cost too much time.
        if self.SIFT:
            index_params = dict(algorithm=1, trees=5)  # Using KD-Tree
        if self.ORB:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)  # check counting

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(features_1, features_2, k=2)
        # Conducting Nearest Neighbor Ratio Test
        good_matches = []
        ratio_thresh = 0.75  # setting threshold
        
        # Best match for m, second for n
        for item in matches:
            if isinstance(item, tuple) and len(item) == 2:
                m, n = item
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        return good_matches

    def find_homography(self, keypoints1, keypoints2, good_matches):
        if len(good_matches) > 4:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # Computing homography and Dealing with outliers: RANSAC
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # Not dealing with outliers
            # H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
            return H, mask
        else:
            return None, None

    def warping(self, H):
        """利用单应性矩阵进行图像拼接"""
        height1, width1 = self.img1.shape[:2]
        height2, width2 = self.img2.shape[:2]
        
        # 计算变换后的边界
        corners = np.array([
            [0, 0], [0, height1], [width1, height1], [width1, 0]
        ], dtype=np.float32).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        [x_min, y_min] = np.int32(transformed_corners.min(axis=0).ravel())
        [x_max, y_max] = np.int32(transformed_corners.max(axis=0).ravel())
        
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
        
        # 进行图像变换
        result = cv2.warpPerspective(self.img1, H_translation @ H, (max(x_max, width2) - x_min, max(y_max, height2) - y_min))

        result[translation_dist[1]:translation_dist[1] + height2, translation_dist[0]:translation_dist[0] + width2] = self.img2
        return result
    
if __name__ == "__main__":
    
    # path1 = 'imgs/part1/stiching_1/CS1.jpg'
    # path2 = 'imgs/part1/stiching_1/CS2.jpg'
    # path1 = 'imgs/part1/stiching_2/UOB1.jpg'
    # path2 = 'imgs/part1/stiching_2/UOB2.jpg'
    path1 = 'imgs/part1/stiching_3/Door1.jpeg'
    path2 = 'imgs/part1/stiching_3/Door2.jpeg'
    stitching = image_stitch(img_path1=path1, img_path2=path2, ORB=True, resize=True)

    (kp1, feats1), (kp2, feats2) = stitching.feature_detect()
    good_matches = stitching.match(features_1=feats1, features_2=feats2)
    H, mask = stitching.find_homography(keypoints1=kp1, keypoints2=kp2, good_matches=good_matches)
    result = stitching.warping(H=H)
    cv2.imwrite('test2.jpg', result)
    # plt.imshow(result)