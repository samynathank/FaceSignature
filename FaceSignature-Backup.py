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

# Load and validate original images
checkin_path = r"C:\Users\skuppuraj\source\repos\Development\POC\FaceSignature\100th.jpg"
test_path = r"C:\Users\skuppuraj\source\repos\Development\POC\FaceSignature\God.jpg"
#checkin_path = ""
#test_path = "" 
checkin_face = load_and_validate_image(checkin_path)
test_face = load_and_validate_image(test_path)

if checkin_face is None or test_face is None:
    print("Exiting due to image loading errors")
    exit()

# Preprocess and save temporary images
temp_checkin_path = preprocess_and_save_image(checkin_face, 'checkin.jpg')
temp_test_path = preprocess_and_save_image(test_face, 'test.jpg')

# Generate face embeddings using temporary preprocessed images
try:
    embeddings_checkin = DeepFace.represent(temp_checkin_path, 
                                          model_name="ArcFace", 
                                          enforce_detection=False)
    if embeddings_checkin:
        print(f"Number of faces detected in check-in image: {len(embeddings_checkin)}")
        for i, face_data in enumerate(embeddings_checkin):
            print(f"Embedding for face {i + 1}: {face_data['embedding']}")
    else:
        print("No faces detected in check-in image.")
except Exception as e:
    print(f"Error processing check-in image: {str(e)}")
    exit()

try:
    embeddings_test = DeepFace.represent(temp_test_path, 
                                       model_name="ArcFace", 
                                       enforce_detection=False)
    if embeddings_test:
        print(f"Number of faces detected in test image: {len(embeddings_test)}")
        for i, face_data in enumerate(embeddings_test):
            print(f"Embedding for test face {i + 1}: {face_data['embedding']}")
    else:
        print("No faces detected in test image.")
except Exception as e:
    print(f"Error processing test image: {str(e)}")
    exit()

# Compute similarities between all check-in and test face embeddings
threshold = 0.6  # Lower threshold for stricter matching
if embeddings_checkin and embeddings_test:
    for i, checkin_embedding in enumerate(embeddings_checkin):
        for j, test_embedding in enumerate(embeddings_test):
            # Try multiple distance metrics
            cosine_similarity = DeepFace.verify(temp_checkin_path, 
                                              temp_test_path,
                                              model_name="VGG-Face",  # Try different model
                                              distance_metric="cosine",
                                              enforce_detection=False)
            
            euclidean_similarity = np.linalg.norm(
                np.array(checkin_embedding['embedding']) - np.array(test_embedding['embedding'])
            )
            
            print(f"Cosine similarity result: {cosine_similarity}")
            print(f"Euclidean similarity: {euclidean_similarity}")
            
            # Combined decision
            if cosine_similarity['verified'] or euclidean_similarity < threshold:
                print(f"Check-in face {i + 1} and test face {j + 1} are the same person.")
            else:
                print(f"Check-in face {i + 1} and test face {j + 1} are different persons.")

# Close all OpenCV windows
cv2.destroyAllWindows()

# Cleanup temporary files
try:
    if os.path.exists(temp_checkin_path):
        os.remove(temp_checkin_path)
        print(f"Deleted temporary file: {temp_checkin_path}")
    if os.path.exists(temp_test_path):
        os.remove(temp_test_path)
        print(f"Deleted temporary file: {temp_test_path}")
except Exception as e:
    print(f"Warning: Could not delete temporary files: {str(e)}")
