import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

from skimage.transform import AffineTransform

#turn [x,y] -> [x,y,1]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


class FeatureExtractor(object):
    GX = 16//2
    GY = 12//2

    def extractRt(self, E):
        W =  np.mat([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)
        u,w,vt = np.linalg.svd(E)
        assert np.linalg.det(u) > 0
        if np.linalg.det(vt) < 0:
            vt *= -1
        # Retrieve the 2 rotations # Change of naming notation ad 2.14
        R = np.dot(np.dot(u,W),vt) # Get the standard rotation matrix
        if np.sum(np.diag(R)) < 0: # If the standard rotation matrix is negative
            R = np.dot(np.dot(u,W.T),vt) # Get the other rotation matrix
        #print("Rotation:", R)
        # Get the translation from essential matrix
        t = u[:,2]
        #print("Translation:", t)
        #print("Vt:", vt)
        pose = np.concatenate([R,t.reshape(3,1)], axis=1)
        return pose

    def __init__(self, K):
        self.orb = cv2.ORB_create(1000) # orb keypoint container
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING2) # Brute force matcher
        #self.bf = cv2.BFMatcher()
        self.last = None
        self.last_frame = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K) 
  
    def normalize(self, pt): # converts pixel coordinates to euclidian coordinates
        return np.dot(self.Kinv, add_ones(pt).T).T[:,0:2] 
    
    def denormalize(self, pt): # converts euclidian coordinates to pixel coordinates
        #print(self.Kinv)
        ret = np.dot(self.K,[pt[0],pt[1],1])
        #print(ret)
        ret /= ret[2] # Normalize by the z coordinate (when F != 1)
        return (int(round(ret[0])), int(round(ret[1])))
    
    def isValid(self, model, data1, data2):
        return model.matrix.shape[0] > 2  & model.matrix.shape[1] > 2

    def extract(self, img, imgcolor):
        # detection
        feats = cv2.goodFeaturesToTrack(img, 3000, qualityLevel=0.01, minDistance=3) # gets features in pixels right
        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats] # converts features to cv2 keypoints
        kps, des = self.orb.compute(img, kps) # computes the descriptors of the keypoints

        # matching
        ret = []
        good = []
        if self.last is not None: # if there is a previous frame
            matches = self.bf.knnMatch(des, self.last['des'], k=2) # Calls bruteforce knn matching with 2 nearest neighbours
            for m,n in matches: # iterate through the matches
                if m.distance < 0.75*n.distance: # Lowe's ratio test, checks if first match is significantly better than the second It is a method to filter!
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))
                    good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        #if self.last is not None: # Nice visualizaton. YOU WERE HERE (VID TIMESTEP 1.54.27)
        #    img3 = cv2.drawMatchesKnn(img,kps,img,self.last['kps'],good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #    cv2.imshow('frame', img3)#,plt.show()
        #    cv2.waitKey(1000)
        # filter
        print("Ret length:", len(ret))
        pose = None
        if len(ret) > 0:
            ret = np.array(ret)
            # Subtract to move to 0
            #ret[:, :, 0] -= img.shape[0]//2
            #ret[:, :, 1] -= img.shape[1]//2
            ret[:, 0, :] = self.normalize(ret[:,0,:])
            ret[:, 1, :] = self.normalize(ret[:,1,:])
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    EssentialMatrixTransform, # less parameters
                                    #FundamentalMatrixTransform, # More paremeters
                                    min_samples=8,
                                    residual_threshold=0.005,
                                    max_trials=100) # We will take this matrix and extract positions
            #print("Inliers:", sum(inliers))  # Print inliers size 
            ret = ret[inliers]
            #print(model.params)
            W =  np.mat([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)

            pose = self.extractRt(model.params)

        # return
        self.last = {'kps': kps, 'des': des}
        self.last_frame = img
        return ret, pose  # Return the matches and the rotation and translation matrix
    
