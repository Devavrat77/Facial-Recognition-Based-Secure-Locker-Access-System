import numpy as np

class FaceComparison:
    # def compare(self, embedding1, embedding2, threshold=0.5):
    #     distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
    #     return distance < threshold
    def compare(self, live_embedding, stored_embedding, threshold=0.8):
        distance = np.linalg.norm(np.array(live_embedding) - np.array(stored_embedding))
        similarity = 1 / (1 + distance)  # Convert distance to similarity score
        print(f"Face similarity score: {similarity:.4f}")
        return similarity > threshold