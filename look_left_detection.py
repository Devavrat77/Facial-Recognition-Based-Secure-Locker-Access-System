# look_left_detection.py

from imutils import face_utils

class LookLeftDetection:
    def __init__(self, shape_predictor, threshold=30):
        self.threshold = threshold
        (self.nStart, self.nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        self.predictor = shape_predictor

    def detect(self, gray, rect):
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        nose = shape[self.nStart:self.nEnd]
        nose_point = nose[3]  # assuming the tip of the nose is at index 3
        return nose_point[0] < self.threshold
