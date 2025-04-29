from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from FaceSignature import load_and_validate_image, preprocess_and_save_image, DeepFace
import numpy as np

app = Flask(__name__)
# Allow requests from any origin
CORS(app, resources={r"/compare-faces": {"origins": "*"}})

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/compare-faces', methods=['POST'])
def compare_faces():
    try:
        # Get file type parameters from form data
        checkin_file_type = request.form.get('checkin_file_type', 'jpg')
        test_file_type = request.form.get('test_file_type', 'jpg')

        # Validate file types
        if checkin_file_type.lower() not in ALLOWED_EXTENSIONS or test_file_type.lower() not in ALLOWED_EXTENSIONS:
            return jsonify({'error': 'Invalid file type specified. Allowed types are: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400

        if 'checkin_image' not in request.files or 'test_image' not in request.files:
            return jsonify({'error': 'Both checkin_image and test_image are required'}), 400

        checkin_file = request.files['checkin_image']
        test_file = request.files['test_image']

        # Validate files with specified types
        if not checkin_file.filename.lower().endswith(checkin_file_type.lower()):
            return jsonify({'error': f'Check-in image must be of type {checkin_file_type}'}), 400
        if not test_file.filename.lower().endswith(test_file_type.lower()):
            return jsonify({'error': f'Test image must be of type {test_file_type}'}), 400

        # Save uploaded files with specified extensions
        checkin_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                  f"checkin_{secure_filename(checkin_file.filename)}.{checkin_file_type}")
        test_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                               f"test_{secure_filename(test_file.filename)}.{test_file_type}")

        checkin_file.save(checkin_path)
        test_file.save(test_path)

        # Process images using FaceSignature.py functions
        checkin_face = load_and_validate_image(checkin_path)  # Calls function from FaceSignature.py
        test_face = load_and_validate_image(test_path)        # Calls function from FaceSignature.py

        if checkin_face is None or test_face is None:
            return jsonify({'error': 'Failed to load images'}), 400

        # Preprocess images using FaceSignature.py function
        temp_checkin_path = preprocess_and_save_image(checkin_face, 'checkin.jpg')  # Calls function from FaceSignature.py
        temp_test_path = preprocess_and_save_image(test_face, 'test.jpg')          # Calls function from FaceSignature.py

        # Uses DeepFace imported from FaceSignature.py
        embeddings_checkin = DeepFace.represent(temp_checkin_path, model_name="ArcFace", enforce_detection=False)
        embeddings_test = DeepFace.represent(temp_test_path, model_name="ArcFace", enforce_detection=False)

        # Compare faces
        results = []
        threshold = 0.6
        
        for i, checkin_embedding in enumerate(embeddings_checkin):
            for j, test_embedding in enumerate(embeddings_test):
                cosine_result = DeepFace.verify(temp_checkin_path, temp_test_path,
                                              model_name="VGG-Face",
                                              distance_metric="cosine",
                                              enforce_detection=False)
                
                euclidean_similarity = float(np.linalg.norm(
                    np.array(checkin_embedding['embedding']) - np.array(test_embedding['embedding'])
                ))

                # Create a serializable result dictionary
                result_dict = {
                    'checkin_face': i + 1,
                    'test_face': j + 1,
                    'cosine_similarity': {
                        'verified': bool(cosine_result['verified']),
                        'distance': float(cosine_result['distance']),
                        'threshold': float(cosine_result['threshold']),
                        'model': str(cosine_result['model']),
                        'detector_backend': str(cosine_result['detector_backend'])
                    },
                    'euclidean_similarity': float(euclidean_similarity),
                    'is_same_person': bool(cosine_result['verified']) or euclidean_similarity < threshold
                }

                results.append(result_dict)

        # Cleanup
        for path in [checkin_path, test_path, temp_checkin_path, temp_test_path]:
            if os.path.exists(path):
                os.remove(path)

        return jsonify({
            'status': 'success',
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on all network interfaces
    app.run(debug=True, host='0.0.0.0', port=5000)
