import cv2
import numpy as np
import time
import PoseEstimationModule as pm 

cap = cv2.VideoCapture("videos/a.mp4")

detector = pm.PoseDetector()

while True:
    #success, img = cap.read()
    #img = cv2.resize(img, (1280,720))
    img = cv2.imread("images/image1.jpg")
    img = detector.findPose(img)
    lmList = detector.getPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
        detector.findAngle(img, 12, 14, 16)

    cv2.imshow("img", img)
    cv2.waitKey(1)
    
