import cv2
import numpy as np
from deepface import DeepFace
import os
from mtcnn import MTCNN

def crop_face(image):
    # Initialize MTCNN detector
    detector = MTCNN()
    
    # Detect faces
    faces = detector.detect_faces(image)
    
    if not faces:
        print("No face detected in image")
        return image
    
    # Get the face with highest confidence
    face = max(faces, key=lambda x: x['confidence'])
    x, y, w, h = face['box']
    
    # Add padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(w + 2*padding, image.shape[1] - x)
    h = min(h + 2*padding, image.shape[0] - y)
    
    # Crop and align face using landmarks
    face_crop = image[y:y+h, x:x+w]
    
    # Display detected face
    cv2.imshow('Detected Face', face_crop)
    cv2.waitKey(1000)  # Show for 1 second
    
    return face_crop

def preprocess_and_save_image(image, filename):
    # Crop face first
    face_crop = crop_face(image)
    # Save temporary file (keeping color for better recognition)
    temp_path = os.path.join(os.path.dirname(__file__), f'temp_{filename}')
    cv2.imwrite(temp_path, face_crop)
    return temp_path

def load_and_validate_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image at {image_path}")
        return None
        
    return img
