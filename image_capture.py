import cv2
import numpy as np
from face_recognition_model import FaceRecognitionModel

class ImageCapture:
    def __init__(self):
        self.face_model = FaceRecognitionModel()

    def capture_images(self, num_images=10):
        cap = cv2.VideoCapture(0)
        images = []
        count = 0

        print("Press 'c' to capture an image when a face is detected. Press 'q' to quit.")

        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face and draw bounding box
            face_rects = self.face_model.detector(frame_rgb)
            for rect in face_rects:
                (x, y, w, h) = (rect.left(), rect.top(), rect.right(), rect.bottom())
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Capture Images', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and face_rects:
                images.append(frame_rgb)
                count += 1
                print(f"Image {count} captured. {num_images - count} remaining.")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return images

    def get_embeddings(self, images):
        return self.face_model.get_embeddings(images)