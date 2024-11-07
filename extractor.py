import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

from skimage.transform import AffineTransform

#turn [x,y] -> [x,y,1]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

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

def normalize(Kinv, pt): # converts pixel coordinates to euclidian coordinates
    return np.dot(Kinv, add_ones(pt).T).T[:,0:2] 

def denormalize(K, pt): # converts euclidian coordinates to pixel coordinates
    ret = np.dot(K,[pt[0],pt[1],1])
    ret /= ret[2] # Normalize by the z coordinate (when F != 1)
    return (int(round(ret[0])), int(round(ret[1])))


def extract(img):
    orb = cv2.ORB_create(1000) # orb keypoint container
    # detection
    pts = cv2.goodFeaturesToTrack(img, 3000, qualityLevel=0.01, minDistance=3) # gets features in pixels right
    # extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts] # converts features to cv2 keypoints
    kps, des = orb.compute(img, kps) # computes the descriptors of the keypoints
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


def getPoseFromRet(ret, Kinv):
    ret[:, 0, :] = normalize(Kinv, ret[:,0,:])
    ret[:, 1, :] = normalize(Kinv, ret[:,1,:])
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            EssentialMatrixTransform, # less parameters
                            #FundamentalMatrixTransform, # More paremeters
                            min_samples=8,
                            residual_threshold=0.005,
                            max_trials=100) # We will take this matrix and extract positions
    #ignore outliers




    ret = ret[inliers]
    #print(model.params) 
    Rt = extractRt(model.params)
    return Rt

def match(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2) # Calls bruteforce knn matching with 2 nearest neighbours
    
    # matching
    ret = []
    good = []
    # Lowes ratio test
    for m,n in matches: # iterate through the matches
        if m.distance < 0.75*n.distance: # Lowe's ratio test, checks if first match is significantly better than the second It is a method to filter!
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            ret.append((p1, p2))
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    #if self.last is not None: # Nice visualizaton. YOU WERE HERE (VID TIMESTEP 1.54.27)
    #    img3 = cv2.drawMatchesKnn(img,kps,img,self.last['kps'],good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #    cv2.imshow('frame', img3)#,plt.show()
    #    cv2.waitKey(1000)
    # filter
    #print("Ret length:", len(ret))
    pose = None
    assert len(ret) >= 8
    ret = np.array(ret)
    return ret



class Frame(object):
    def __init__(self, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K) 
        
        self.pts, self.des = extract(img)
        self.pts = normalize(self.Kinv, self.pts)
        pass
      
    def isValid(self, model, data1, data2):
        return model.matrix.shape[0] > 2  & model.matrix.shape[1] > 2