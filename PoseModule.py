import cv2 as cv
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detection_con=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detection_con = detection_con
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.results = None  # Added line to store results

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results is not None and self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img


    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (0, 125, 0), thickness=-1)

        return lmList




def main():
        
        cap = cv.VideoCapture('Workout.mp4')
        pTime = 0
        detector = poseDetector()

        while True:
            success, img = cap.read()
            img = detector.findPose(img)
            lmList = detector.getPosition(img, draw = False)
            print(lmList[14])
            cv.circle(img, (lmList[14][1],lmList[14][2]), 8, (144,0,0), thickness=-1)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_COMPLEX,3, (144,0,0), thickness=3 )
            cv.imshow('Video', img)

            if cv.waitKey(1) == ord ('a'):
                return
            cv.waitKey(1)


if __name__ == "__main__":
    main()