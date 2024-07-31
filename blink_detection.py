# # blink_detection.py
# ---------------------------------------------------------working fine ----------------------------------------------------
# # import the necessary packages
# from scipy.spatial import distance as dist
# from imutils.video import VideoStream
# from imutils import face_utils
# import numpy as np
# import imutils
# import dlib
# import time
# import cv2

# class BlinkDetector:
#     def __init__(self, shape_predictor_path, eye_ar_thresh=0.3, eye_ar_consec_frames=3):
#         self.EYE_AR_THRESH = eye_ar_thresh
#         self.EYE_AR_CONSEC_FRAMES = eye_ar_consec_frames
#         self.COUNTER = 0
#         self.TOTAL = 0

#         # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
#         print("[INFO] loading facial landmark predictor...")
#         self.detector = dlib.get_frontal_face_detector()
#         self.predictor = dlib.shape_predictor(shape_predictor_path)

#         # grab the indexes of the facial landmarks for the left and right eye, respectively
#         (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#         (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#         # start the video stream thread
#         print("[INFO] starting video stream...")
#         self.vs = VideoStream(src=0).start()
#         time.sleep(1.0)

#     def eye_aspect_ratio(self, eye):
#         A = dist.euclidean(eye[1], eye[5])
#         B = dist.euclidean(eye[2], eye[4])
#         C = dist.euclidean(eye[0], eye[3])
#         ear = (A + B) / (2.0 * C)
#         return ear

#     def get_ear_from_frame(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = self.detector(gray, 0)
#         for rect in rects:
#             shape = self.predictor(gray, rect)
#             shape = face_utils.shape_to_np(shape)
#             leftEye = shape[self.lStart:self.lEnd]
#             rightEye = shape[self.rStart:self.rEnd]
#             leftEAR = self.eye_aspect_ratio(leftEye)
#             rightEAR = self.eye_aspect_ratio(rightEye)
#             ear = (leftEAR + rightEAR) / 2.0
#             return ear
#         return None

#     def detect_blinks(self):
#         while True:
#             frame = self.vs.read()
#             frame = imutils.resize(frame, width=450)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             rects = self.detector(gray, 0)

#             for rect in rects:
#                 shape = self.predictor(gray, rect)
#                 shape = face_utils.shape_to_np(shape)
#                 leftEye = shape[self.lStart:self.lEnd]
#                 rightEye = shape[self.rStart:self.rEnd]
#                 leftEAR = self.eye_aspect_ratio(leftEye)
#                 rightEAR = self.eye_aspect_ratio(rightEye)
#                 ear = (leftEAR + rightEAR) / 2.0

#                 leftEyeHull = cv2.convexHull(leftEye)
#                 rightEyeHull = cv2.convexHull(rightEye)
#                 cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#                 cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

#                 if ear < self.EYE_AR_THRESH:
#                     self.COUNTER += 1
#                 else:
#                     if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
#                         self.TOTAL += 1
#                     self.COUNTER = 0

#                 cv2.putText(frame, "Blinks: {}".format(self.TOTAL), (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#             cv2.imshow("Frame", frame)
#             key = cv2.waitKey(1) & 0xFF

#             if key == ord("q"):
#                 break

#         cv2.destroyAllWindows()
#         self.vs.stop()
# if __name__ == "__main__":
#     blink_detection = BlinkDetector("shape_predictor_68_face_landmarks.dat")
#     blink_detection.detect_blinks()



# blink_detection.py

# from scipy.spatial import distance as dist
# from imutils import face_utils
# import numpy as np
# import cv2

# class BlinkDetection:
#     def __init__(self, shape_predictor, eye_ar_thresh=0.2, eye_ar_consec_frames=3):
#         self.EYE_AR_THRESH = eye_ar_thresh
#         self.EYE_AR_CONSEC_FRAMES = eye_ar_consec_frames
#         self.COUNTER = 0
#         self.TOTAL = 0

#         (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#         (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#         self.predictor = shape_predictor

#     def eye_aspect_ratio(self, eye):
#         A = dist.euclidean(eye[1], eye[5])
#         B = dist.euclidean(eye[2], eye[4])
#         C = dist.euclidean(eye[0], eye[3])
#         ear = (A + B) / (2.0 * C)
#         return ear

#     def detect(self, gray, rect):
#         shape = self.predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)

#         leftEye = shape[self.lStart:self.lEnd]
#         rightEye = shape[self.rStart:self.rEnd]
#         leftEAR = self.eye_aspect_ratio(leftEye)
#         rightEAR = self.eye_aspect_ratio(rightEye)
#         ear = (leftEAR + rightEAR) / 2.0

#         if ear < self.EYE_AR_THRESH:
#             self.COUNTER += 1
#         else:
#             if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
#                 self.TOTAL += 1
#             self.COUNTER = 0

#         return ear, self.TOTAL


# if __name__ == "__main__":
#     blink_detection = BlinkDetection("shape_predictor_68_face_landmarks.dat")
#     blink_detection.detect()

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import cv2
import dlib

class BlinkDetection:
    def __init__(self, shape_predictor_path, eye_ar_thresh=0.2, eye_ar_consec_frames=3):
        self.EYE_AR_THRESH = eye_ar_thresh
        self.EYE_AR_CONSEC_FRAMES = eye_ar_consec_frames
        self.COUNTER = 0
        self.TOTAL = 0

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.predictor = dlib.shape_predictor(shape_predictor_path)

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect(self, gray, rect):
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

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

# if __name__ == "__main__":
    # Load the shape predictor
    # shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
    # blink_detection = BlinkDetection(shape_predictor_path)

    # # Initialize the webcam
    # cap = cv2.VideoCapture(0)
    # detector = dlib.get_frontal_face_detector()

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     rects = detector(gray)

    #     for rect in rects:
    #         ear, total = blink_detection.detect(gray, rect)
    #         cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #         cv2.putText(frame, f"Blinks: {total}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #     cv2.imshow("Blink Detection", frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
