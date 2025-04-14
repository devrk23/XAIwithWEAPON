import requests
import json
import time
from datetime import datetime

API_BASE_URL = 'http://localhost:5000'

def test_health():
    """Test health check endpoint."""
    print("\n=== Testing Health Check ===")
    response = requests.get(f'{API_BASE_URL}/api/health')
    print(json.dumps(response.json(), indent=2))

def test_model_info():
    """Test model info endpoint."""
    print("\n=== Testing Model Info ===")
    response = requests.get(f'{API_BASE_URL}/api/model/info')
    print(json.dumps(response.json(), indent=2))

def test_image_detection():
    """Test image detection endpoint."""
    print("\n=== Testing Image Detection ===")
    image_path = "model/guns.jpg"  # Update this path to your test image
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f'{API_BASE_URL}/api/detect/image', files=files)
            result = response.json()
            
            # Print basic info and first detection
            print(f"Status: {result['status']}")
            print(f"Timestamp: {result['timestamp']}")
            if 'detections' in result and result['detections']:
                print("\nFirst Detection:")
                print(json.dumps(result['detections'][0], indent=2))
            print(f"\nResult saved to: {result.get('result_file')}")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_video_detection():
    """Test video detection endpoint."""
    print("\n=== Testing Video Detection ===")
    video_path = "model/test video.mp4"  # Update this path to your test video
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            response = requests.post(f'{API_BASE_URL}/api/detect/video', files=files)
            result = response.json()
            
            # Print basic info and first frame detection
            print(f"Status: {result['status']}")
            print(f"Timestamp: {result['timestamp']}")
            if 'video_info' in result:
                print("\nVideo Info:")
                print(json.dumps(result['video_info'], indent=2))
            if 'detections' in result and result['detections']:
                print("\nFirst Frame Detection:")
                print(json.dumps(result['detections'][0], indent=2))
            print(f"\nResult saved to: {result.get('result_file')}")
    except Exception as e:
        print(f"Error: {str(e)}")

def test_stats():
    """Test statistics endpoint."""
    print("\n=== Testing Statistics ===")
    response = requests.get(f'{API_BASE_URL}/api/stats')
    print(json.dumps(response.json(), indent=2))

def test_results():
    """Test results history endpoint."""
    print("\n=== Testing Results History ===")
    response = requests.get(f'{API_BASE_URL}/api/results')
    print(json.dumps(response.json(), indent=2))

def run_all_tests():
    """Run all API tests."""
    print("Starting API Tests...")
    print(f"Time: {datetime.now().isoformat()}")
    print("API URL:", API_BASE_URL)
    
    # Run tests
    test_health()
    test_model_info()
    test_image_detection()
    test_video_detection()
    test_stats()
    test_results()
    
    print("\nAPI Tests Completed!")

if __name__ == "__main__":
    run_all_tests() 