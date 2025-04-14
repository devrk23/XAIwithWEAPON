from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from datetime import datetime
from model.detecting_images import detector, detect_objects_in_photo, detect_objects_in_video
from model.realtime_xai import RealtimeXAI
import base64
import json
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'detection_results'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# Store detection statistics
detection_stats = {
    'total_detections': 0,
    'image_detections': 0,
    'video_detections': 0,
    'realtime_detections': 0,
    'detected_weapons': {
        'Guns': 0,
        'Knives': 0
    }
}

def update_stats(detections, detection_type):
    """Update detection statistics."""
    detection_stats[f'{detection_type}_detections'] += 1
    detection_stats['total_detections'] += 1
    
    if isinstance(detections, list):
        for detection in detections:
            if 'class' in detection:
                detection_stats['detected_weapons'][detection['class']] += 1
    elif isinstance(detections, dict) and 'frames' in detections:
        for frame in detections['frames']:
            if 'detections' in frame:
                for detection in frame['detections']:
                    if 'class' in detection:
                        detection_stats['detected_weapons'][detection['class']] += 1

def save_detection_result(result, detection_type):
    """Save detection result to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{detection_type}_{timestamp}.json"
    filepath = os.path.join(RESULTS_FOLDER, filename)
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=4)
    
    return filepath

def save_uploaded_file(file):
    """Save uploaded file and return the path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath

def encode_image_to_base64(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """Endpoint for image detection."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save uploaded file
        image_path = save_uploaded_file(file)
        
        # Process image
        detections = detect_objects_in_photo(image_path, return_json=True)
        
        # Get the processed image with detections
        processed_image_path = "./imgs/Test/teste.jpg"
        
        # Convert processed image to base64
        if os.path.exists(processed_image_path):
            processed_image_base64 = encode_image_to_base64(processed_image_path)
        else:
            processed_image_base64 = None
        
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'processed_image': processed_image_base64
        }
        
        # Update statistics
        update_stats(detections, 'image')
        
        # Save result
        result_file = save_detection_result(response, 'image')
        response['result_file'] = result_file
        
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    """Endpoint for video detection."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Save uploaded file
        video_path = save_uploaded_file(file)
        
        # Process video
        result = detect_objects_in_video(video_path, return_json=True)
        
        # Convert output video to base64 if needed
        output_video_path = result['output_path']
        if os.path.exists(output_video_path):
            with open(output_video_path, "rb") as video_file:
                video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        else:
            video_base64 = None
        
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'video_info': result['video_info'],
            'detections': result['frames'],
            'processed_video': video_base64
        }
        
        # Update statistics
        update_stats(result, 'video')
        
        # Save result
        result_file = save_detection_result(response, 'video')
        response['result_file'] = result_file
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/detect/realtime', methods=['POST'])
def detect_realtime():
    """Endpoint for real-time detection from webcam stream."""
    try:
        # Get frame from request
        frame_data = request.json.get('frame')
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Convert base64 frame to image
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        output_frame, explanations = detector.xai_system.process_frame(frame, return_json=True)
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', output_frame)
        processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'detections': explanations,
            'processed_frame': processed_frame_base64
        }
        
        # Update statistics
        update_stats(explanations, 'realtime')
        
        # Save result
        result_file = save_detection_result(response, 'realtime')
        response['result_file'] = result_file
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get information about the model and its capabilities."""
    try:
        model_info = {
            'name': 'Weapon Detection XAI Model',
            'version': '1.0',
            'supported_classes': ['Guns', 'Knives'],
            'features': {
                'edge_analysis': True,
                'shape_analysis': True,
                'feature_detection': True,
                'confidence_scoring': True,
                'heatmap_visualization': True,
                'real_time_processing': True
            },
            'input_formats': {
                'image': ['jpg', 'jpeg', 'png'],
                'video': ['mp4', 'avi']
            },
            'xai_capabilities': {
                'feature_explanations': True,
                'confidence_analysis': True,
                'threat_assessment': True,
                'visual_explanations': True
            }
        }
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'model_info': model_info
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get detection statistics."""
    try:
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'statistics': detection_stats
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get list of saved detection results."""
    try:
        results = []
        for filename in os.listdir(RESULTS_FOLDER):
            if filename.endswith('.json'):
                filepath = os.path.join(RESULTS_FOLDER, filename)
                with open(filepath, 'r') as f:
                    result = json.load(f)
                results.append({
                    'filename': filename,
                    'timestamp': result.get('timestamp'),
                    'type': filename.split('_')[0],
                    'detections': len(result.get('detections', []))
                })
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'results': sorted(results, key=lambda x: x['timestamp'], reverse=True)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# Test health check
response = requests.get('http://localhost:5000/api/health')
print(response.json())

# Test image detection
image_path = "model/guns.jpg"  # Use one of your test images
with open(image_path, 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/api/detect/image', files=files)
print(response.json())

# Test video detection
video_path = "model/test video.mp4"  # Use your test video
with open(video_path, 'rb') as f:
    files = {'video': f}
    response = requests.post('http://localhost:5000/api/detect/video', files=files)
print(response.json())

# Test model info
response = requests.get('http://localhost:5000/api/model/info')
print(response.json())

# Test statistics
response = requests.get('http://localhost:5000/api/stats')
print(response.json())

# Test results history
response = requests.get('http://localhost:5000/api/results')
print(response.json()) 