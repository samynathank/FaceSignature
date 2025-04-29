# Face Signature Comparison Documentation

## Import Statements
# Run this command before start this web api 'pip install -r requirements.txt'
```python
import cv2  # OpenCV library for image processing
import numpy as np  # NumPy for numerical operations
from deepface import DeepFace  # DeepFace for face recognition
import os  # Operating system operations
from mtcnn import MTCNN  # MTCNN for face detection
```

## Functions

### 1. crop_face(image)
Purpose: Detects and crops faces from input images
```python
def crop_face(image):
    # Initialize MTCNN detector for face detection
    detector = MTCNN()
    
    # Detect faces in the image
    faces = detector.detect_faces(image)
    
    # If no faces found, return original image
    if not faces:
        print("No face detected in image")
        return image
    
    # Select face with highest confidence
    face = max(faces, key=lambda x: x['confidence'])
    x, y, w, h = face['box']  # Get face coordinates
    
    # Add padding around face
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(w + 2*padding, image.shape[1] - x)
    h = min(h + 2*padding, image.shape[0] - y)
    
    # Crop face region
    face_crop = image[y:y+h, x:x+w]
    
    # Display cropped face for verification
    cv2.imshow('Detected Face', face_crop)
    cv2.waitKey(1000)
    
    return face_crop
```

### 2. preprocess_and_save_image(image, filename)
Purpose: Preprocesses and saves temporary face images
```python
def preprocess_and_save_image(image, filename):
    face_crop = crop_face(image)  # Crop face
    temp_path = os.path.join(os.path.dirname(__file__), f'temp_{filename}')  # Create temp file path
    cv2.imwrite(temp_path, face_crop)  # Save image
    return temp_path
```

### 3. load_and_validate_image(image_path)
Purpose: Loads and validates image files
```python
def load_and_validate_image(image_path):
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image at {image_path}")
        return None
        
    return img
```
