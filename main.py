import os
import json
from datetime import datetime
from model.detecting_images import detect_objects_and_plot, detect_objects_in_video
from model.realtime_xai import RealtimeXAI

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu():
    """Display the main menu."""
    print("\n=== Weapon Detection System ===")
    print("1. Image Detection")
    print("2. Video Detection")
    print("3. Real-time Detection")
    print("4. Exit")
    print("\nPlease enter your choice (1-4): ")

def save_json_output(data, mode):
    """Save detection results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "detection_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"{output_dir}/detection_{mode}_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    return filename

def image_detection():
    """Handle image detection mode."""
    clear_screen()
    print("=== Image Detection Mode ===")
    print("Please enter the path to your image file (e.g., imgs/test.jpg)")
    image_path = input().strip()
    
    if not os.path.exists(image_path):
        result = {
            "status": "error",
            "message": f"File '{image_path}' not found!",
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(result, indent=4))
        input("\nPress Enter to continue...")
        return
    
    print("\nProcessing image...")
    try:
        detections = detect_objects_and_plot(image_path, return_json=True)
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "file_path": image_path,
            "detections": detections
        }
        
        # Save and display results
        output_file = save_json_output(result, "image")
        print(f"\nResults saved to: {output_file}")
        print("\nDetection Results:")
        print(json.dumps(result, indent=4))
        print("\nPress any key in the image window to continue.")
    except Exception as e:
        result = {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(result, indent=4))
        input("\nPress Enter to continue...")

def video_detection():
    """Handle video detection mode."""
    clear_screen()
    print("=== Video Detection Mode ===")
    print("Please enter the path to your video file (e.g., videos/test.mp4)")
    video_path = input().strip()
    
    if not os.path.exists(video_path):
        result = {
            "status": "error",
            "message": f"File '{video_path}' not found!",
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(result, indent=4))
        input("\nPress Enter to continue...")
        return
    
    print("\nStarting video detection...")
    print("Press 'q' to quit the video window")
    
    try:
        detections = detect_objects_in_video(video_path, return_json=True)
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "file_path": video_path,
            "output_video": detections["output_path"],
            "detections": detections["frames"]
        }
        
        # Save and display results
        output_file = save_json_output(result, "video")
        print(f"\nResults saved to: {output_file}")
        print("\nDetection Results:")
        print(json.dumps(result, indent=4))
        input("\nPress Enter to continue...")
    except Exception as e:
        result = {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(result, indent=4))
        input("\nPress Enter to continue...")

def realtime_detection():
    """Handle real-time detection mode."""
    clear_screen()
    print("=== Real-time Detection Mode ===")
    print("Initializing real-time detection system...")
    
    try:
        # Initialize XAI system with correct model path
        model_path = 'model/yolov8n.pt'
        xai_system = RealtimeXAI(model_path)
        
        print("\nStarting real-time detection...")
        print("Press 'q' to quit the video window")
        
        # Start real-time detection with JSON output
        detections = xai_system.start_realtime_detection(return_json=True)
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "session_duration": detections["session_duration"],
            "total_frames": detections["total_frames"],
            "detections": detections["frames"]
        }
        
        # Save and display results
        output_file = save_json_output(result, "realtime")
        print(f"\nResults saved to: {output_file}")
        print("\nDetection Results:")
        print(json.dumps(result, indent=4))
        input("\nPress Enter to continue...")
    except Exception as e:
        result = {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        print(json.dumps(result, indent=4))
        input("\nPress Enter to continue...")

def main():
    """Main program loop."""
    while True:
        clear_screen()
        print_menu()
        
        choice = input().strip()
        
        if choice == '1':
            image_detection()
        elif choice == '2':
            video_detection()
        elif choice == '3':
            realtime_detection()
        elif choice == '4':
            result = {
                "status": "exit",
                "message": "Thank you for using the Weapon Detection System!",
                "timestamp": datetime.now().isoformat()
            }
            print(json.dumps(result, indent=4))
            break
        else:
            result = {
                "status": "error",
                "message": "Invalid choice! Please try again.",
                "timestamp": datetime.now().isoformat()
            }
            print(json.dumps(result, indent=4))
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 