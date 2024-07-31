from customer import Customer
from liveness_detection import LivenessDetection
from image_capture import ImageCapture
from face_recognition_model import FaceRecognitionModel
from face_comparison import FaceComparison
from security_key import SecurityKey
from access_locker import AccessLocker

def main():
    customer = Customer(customer_id="1111", locker_number="11")
    
    if LivenessDetection("shape_predictor_68_face_landmarks.dat").start_detection():
    # if True:
        images = ImageCapture().capture_images()
        model = FaceRecognitionModel()
        embeddings = model.get_embeddings(images)
        if not embeddings:
            print("No valid face embeddings detected. Please try again.")
            return
        avg_embedding = model.average_embedding(embeddings)
        print("Average Embedding : " ,avg_embedding)
        
        try:
            stored_embedding = customer.get_stored_embedding()
            print("Stored Embedding : ",stored_embedding)
        except ValueError as e:
            print(f"Error: {str(e)}")
            return
        # stored_embedding = customer.get_stored_embedding()
        if FaceComparison().compare(avg_embedding, stored_embedding):
            # key = SecurityKey().generate()
            # AccessLocker().unlock(customer.locker_number, key)
            print("Access Granted.")
        else:
            print("Access Denied, Face not recognized.")
    else:
        print("Liveness Detection Failed")

if __name__ == "__main__":
    main()