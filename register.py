import cv2
import numpy as np
import os
from face_recognition_model import FaceRecognitionModel
from customer import Customer

def capture_and_select_frames(customer_id):
    cap = cv2.VideoCapture(0)
    saved_frames = []
    face_rects =[]
    frame_count = 0
    
    model = FaceRecognitionModel()
    
    if not os.path.exists(customer_id):
        os.makedirs(customer_id)
    
    print("Press 's' to attempt to save the current frame, 'q' to quit and finish.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face and draw landmarks
        rects = model.detector(frame_rgb)
        frame_with_drawings = frame.copy()  # Create a copy for drawings
        
        for rect in rects:
            shape = model.predictor(frame_rgb, rect)
            for i in range(68):
                x, y = shape.part(i).x, shape.part(i).y
                cv2.circle(frame_with_drawings, (x, y), 2, (0, 255, 0), -1)

        if rects:
            for rect in rects:
                (x, y, w, h) = (rect.left(), rect.top(), rect.right(), rect.bottom())
                cv2.rectangle(frame_with_drawings, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame_with_drawings, "Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame_with_drawings, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Capture Frames", frame_with_drawings)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if rects:
                filename = os.path.join(customer_id, f"frame_{frame_count}.jpg")
                cv2.imwrite(filename, frame)  # Save original frame
                saved_frames.append(filename)
                face_rects.append(rects[0])  # Save the first detected face rectangle
                print(f"Saved frame {filename}")
                frame_count += 1
            else:
                print("No face detected. Frame not saved.")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return saved_frames , face_rects


def get_face_embeddings(image_paths, face_rects, model):
    embeddings = []
    for path, rect in zip(image_paths, face_rects):
        image = cv2.imread(path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape = model.predictor(image_rgb, rect)
        embedding = model.face_recognizer.compute_face_descriptor(image_rgb, shape)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            print(f"Failed to get embedding for image: {path}")
    print("-----" ,embeddings)
    return embeddings

def get_image_paths(folder):
    image_paths = []
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(folder, file_name))
    return image_paths


def register_customer(customer_id, locker_number, embeddings):
    """Registers a new customer with average face embedding."""
    model = FaceRecognitionModel()
    
    # Ensure at least one valid embedding exists
    if not embeddings:
        print("No valid embeddings found. Registration failed.")
        return

    # Compute average embedding
    avg_embedding = model.average_embedding(embeddings)
    
    # Register customer with average embedding
    customer = Customer(customer_id, locker_number)
    customer.add_customer(avg_embedding)
    
    print(f"Customer {customer_id} registered successfully with locker {locker_number}")

if __name__ == "__main__":
    customer_id = input("Enter customer ID: ")
    locker_number = input("Enter locker number: ")
    
    print("Capturing frames. Press 's' to save frames with bounding boxes and 'q' to finish.")
    image_paths , face_rects = capture_and_select_frames(customer_id)
    # image_paths = get_image_paths(r"D:\Programming\Cogni_proj\Project_SLO\125125")
    
    if len(image_paths) < 5:
        print("Please capture at least 5 frames.")
    else:
        print("Computing face embeddings...")
        model = FaceRecognitionModel()
        embeddings = get_face_embeddings(image_paths,face_rects, model)
        
        if not embeddings:
            print("No valid embeddings were generated. Please try again.")
        else:
            if len(embeddings) > 10:
                embeddings = embeddings[:10]  # Limit to 10 images
            
            try:
                register_customer(customer_id, locker_number, embeddings)
                print(f"Successfully registered customer {customer_id} with {len(embeddings)} face embeddings.")
            except Exception as e:
                print(f"Failed to register customer: {str(e)}")