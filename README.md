# Facial-Recognition-Based-Secure-Locker-Access-System

## Overview
This project simplifies the operation of bank lockers by using face recognition technology to unlock the lockers. The system ensures security through liveness detection and face comparison. It captures multiple images or a short video clip of the user, computes their face embeddings, and compares them with stored embeddings to grant access.

## Project Structure
### face_recognition_model.py: Contains the FaceRecognitionModel class for face recognition and embedding extraction.
### image_capture.py: Handles capturing images from the webcam.
### liveness_detection.py: Implements liveness detection using blink detection and head movements.
### main.py: Main script that integrates all components for the locker operation process.
### register.py: Script for registering a new customer by capturing images and storing their face embeddings.
### create_db.py: Script to create the SQLite database for storing customer data.
### customer.py: Contains the Customer class for managing customer information and face embeddings.
### face_comparison.py: Implements the face comparison logic.
### security_key.py and access_locker.py: (Assumed to be part of the project based on flowchart, but not provided) These would handle security key generation and locker access.

#### git clone <repository-url>
#### cd <repository-name>

### Install dependencies:
#### Install the required Python packages using pip.

#### pip install -r requirements.txt

### Download required models:

### dlib_face_recognition_resnet_model_v1.dat.bz2
shape_predictor_68_face_landmarks.dat
Place these files in the project directory and ensure they are decompressed if necessary.

### Create the database:
Run the following command to create the SQLite database:

### python create_db.py
Usage
Register a Customer
Run the register.py script to capture images and store the customer's face embeddings:

### python register.py
Follow the prompts to enter the customer ID and locker number. Capture the required number of images.
Access the Locker
Run the main.py script to start the locker access process:

### python main.py
The system will perform liveness detection, capture images, compute embeddings, and compare them with stored embeddings. If successful, a security key will be generated and the locker will be accessed.
Liveness Detection
The liveness detection process includes:

### Blink detection
Head movements (left and right)
The system ensures that only live individuals (not photos or videos) can access the locker.

### Face Recognition
The face recognition process involves:

Capturing multiple images or a short video clip.
Extracting face embeddings using a pre-trained model (FaceNet or ArcFace).
Averaging the embeddings to create a single vector.
Comparing the live embeddings with stored embeddings using Euclidean distance or cosine similarity.
Security and Access
A security key is generated upon successful face recognition and liveness detection. This key is validated for a few minutes to grant access to the locker.
