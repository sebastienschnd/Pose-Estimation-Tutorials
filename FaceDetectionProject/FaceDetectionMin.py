import cv2
import time
import mediapipe as mp 

cap = cv2.VideoCapture("videos/b.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results =  faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            print(id, detection)
            print(detection.score)
            print(detection.location_data.relative_bounding_box)
   
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    
    cv2.imshow("Image", img)

    cv2.waitKey(1)