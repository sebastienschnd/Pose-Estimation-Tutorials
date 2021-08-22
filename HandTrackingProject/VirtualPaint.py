import cv2
import numpy as np 
import time
import os
import HandTrackingModule as htm 

folderPath = "Brushes"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))   
header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = htm.handDetector(detectionCon=0.85)

while True:

    # 1. import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


    # 3. Check which fingers are up
    # 4. If selection mode - Two finger are up
    # 5. If drawing Mode - Index finger is up

    # Setting the header image
    img[0:125, 0:640] = header[0:125, 0:640]
    cv2.imshow("image", img)
    cv2.waitKey(1)