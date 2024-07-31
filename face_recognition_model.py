import bz2
import shutil
import dlib
import numpy as np
import cv2

class FaceRecognitionModel:
    def __init__(self):
        self._decompress_model()
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()

    def _decompress_model(self):
        with bz2.open("dlib_face_recognition_resnet_model_v1.dat.bz2", "rb") as source_file, open("dlib_face_recognition_resnet_model_v1.dat", "wb") as dest_file:
            shutil.copyfileobj(source_file, dest_file)

    def get_embeddings(self, images):
        return [self._get_embedding(image) for image in images]

    def _get_embedding(self, image):
        face_rects = self.detector(image)
        if face_rects:
            shape = self.predictor(image, face_rects[0])
            # print("-------------embedding-------------")
            # print(self.face_recognizer.compute_face_descriptor(image, shape))
            return self.face_recognizer.compute_face_descriptor(image, shape)
        print("No face detected in the image.")
        return None

    def average_embedding(self, embeddings):
        return np.mean(embeddings, axis=0)

    def draw_bounding_boxes(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        face_rects = self.detector(gray)
        
        for rect in face_rects:
            (x, y, w, h) = (rect.left(), rect.top(), rect.right(), rect.bottom())
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
        
        return image
