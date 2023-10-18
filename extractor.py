import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform



class FeautureExtractor(object):
    GX = 16//2
    GY = 12//2
    def __init__(self):
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING )
        #self.bf = cv2.BFMatcher()
        self.last = None
        self.seed = 9
        self.rng = np.random.default_rng(self.seed)
        
    def extract(self, img):
        # Detection 
        feats =  cv2.goodFeaturesToTrack(img, 3000, qualityLevel=0.01, minDistance=3)
        # Encoding
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        # Matching
        ret = []
        if(self.last is not None):
            #matches = self.bf.match(des, self.last['des'])
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m,n in matches: 
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1,kp2)) 
            #matches = zip([kps[m.queryIdx] for m in matches], [self.last['kps'][m.trainIdx] for m in matches]) 

        self.last = {'kps': kps, 'des' : des}
        # filter
        if len(ret) > 0:
            
            ret = np.array(ret)

            tmp = (ret[:,0], ret[:,1])
            print("Ret shape",ret.shape)
            model, inliers = ransac((ret[:,0], ret[:,1]), FundamentalMatrixTransform,
                                    min_samples=8, residual_threshold=0.01, max_trials=100, random_state=self.rng)
        # return
        return ret