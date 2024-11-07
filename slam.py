#/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 python3
import os
import cv2
import numpy as np
import time
from extractor import *
import math

import g2o

# Current time 3.57.07
# Camera intrinsics
W = 1920//2
H = 1080//2
F = 270
K = np.array(([F,0,W//2],[0,F,H//2],[0,0,1]))
orb = cv2.ORB_create()


frames = []
def process_frame(img, frame):
    img = cv2.resize(img, (W,H))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame = Frame(imgray, K)
    frames.append(frame)

    if (len(frames) <= 1 ):
        return
    # else
    matches = match(frames[-1], frames[-2])
    if matches is not None:
        for p1, p2 in matches:
            u1,v1 = denormalize(K, p1)
            u2,v2 = denormalize(K, p2)
            cv2.circle(img, (u1,v1), color=(0,255,0), radius=1g)
            cv2.line(img, (u1,v1), (u2,v2), (255,0,0), thickness=1)
    cv2.imshow("frame",img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()

if __name__ == "__main__":
    cap = cv2.VideoCapture('videos/test_countryroad.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame, "Frame")
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    

    print("Hello")