import cv2 as cv
import mediapipe as mp
import time


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv.VideoCapture('Workout.mp4')
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv.circle(img, (cx,cy), 5, (0,125,0), thickness=-1)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_COMPLEX,3, (144,0,0), thickness=3 )
    cv.imshow('Video', img)

    if cv.waitKey(1) == ord ('a'):
        break
    cv.waitKey(1)