#/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 python3
import os
import cv2
import numpy
import time
from extractor import FeautureExtractor 


W = 1920//2
H = 1080//2

orb = cv2.ORB_create()


fe = FeautureExtractor()
 
def process_frame(img, frame):
    img = cv2.resize(img, (W,H))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #fe.extract(imgray)
    print("Calculating keypoints")
    matches = fe.extract(imgray)
    if matches is not None:
        for p1, p2 in matches:
            u1,v1 = map(lambda x: int(round(x)), p1)
            u2,v2 = map(lambda x: int(round(x)), p2 )
            cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
            cv2.line(img, (u1,v1), (u2,v2), (255,0,0), thickness=2)
    cv2.imshow(frame,img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
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