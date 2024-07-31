#------------------------------------------------------------------------working-------------------------------------------------------------------
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import dlib
import time
import cv2

class LivenessDetection:
    def __init__(self, shape_predictor_path, eye_ar_thresh=0.22, eye_ar_consec_frames=3, success_threshold=5):
        self.EYE_AR_THRESH = eye_ar_thresh
        self.EYE_AR_CONSEC_FRAMES = eye_ar_consec_frames
        self.COUNTER = 0
        self.TOTAL = 0
        self.SUCCESS_THRESHOLD = success_threshold

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.nStart, self.nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

        print("[INFO] starting video stream...")
        self.vs = VideoStream(src=0).start()
        time.sleep(1.0)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_blinks(self, shape):
        leftEye = shape[self.lStart:self.lEnd]
        rightEye = shape[self.rStart:self.rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < self.EYE_AR_THRESH:
            self.COUNTER += 1
        else:
            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                self.TOTAL += 1
            self.COUNTER = 0

        return ear, self.TOTAL

    def detect_look_left(self, shape):
        nose = shape[self.nStart:self.nEnd]
        nose_point = nose[3]  # assuming the tip of the nose is at index 3
        if nose_point[0] < 30:  # Threshold for left look, you might need to adjust this
            return True
        return False

    def detect_look_right(self, shape):
        nose = shape[self.nStart:self.nEnd]
        nose_point = nose[3]  # assuming the tip of the nose is at index 3
        if nose_point[0] > 70:  # Threshold for right look, you might need to adjust this
            return True
        return False

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            ear, blinks = self.detect_blinks(shape)
            is_looking_left = self.detect_look_left(shape)
            is_looking_right = self.detect_look_right(shape)

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Look Left: {}".format(is_looking_left), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Look Right: {}".format(is_looking_right), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def start_detection(self):
        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=450)
            frame = self.process_frame(frame)

            # Display the updated blink counter and EAR on the frame
            cv2.putText(frame, f"Total Blinks: {self.TOTAL}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Check if the blink count meets or exceeds the success threshold
            if self.TOTAL >= self.SUCCESS_THRESHOLD:
                print("Liveness Detection Successful")
                cv2.putText(frame, "Liveness Detection Successful", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Frame", frame)
                cv2.waitKey(3000)  # Display success message for 3 seconds
                break

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        self.vs.stop()
        return True

if __name__ == "__main__":
    liveness_detection = LivenessDetection("shape_predictor_68_face_landmarks.dat")
    result = liveness_detection.start_detection()
    print(f"Detection Result: {result}")
