import cv2
import numpy as np
import os
import time
from cvzone.HandTrackingModule import HandDetector

class HandGestureZoom:
    def __init__(self, camera_index=None):
        # Camera setup
        self.camera_index = self.select_camera() if camera_index is None else camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify camera connection
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            raise RuntimeError(f"Camera {self.camera_index} not accessible")
        
        # Hand detector setup
        self.detector = HandDetector(detectionCon=0.7, maxHands=1)
        
        # Zoom parameters
        self.startDis = None
        self.scale = 0
        self.cx, self.cy = 200, 200
        self.min_scale = -200
        self.max_scale = 300
        
        # Image management
        self.current_image_index = 0
        self.images = self.load_images()
        self.current_image = None
        
        # UI elements
        self.show_instructions = True
        self.show_fps = True
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Gesture states
        self.gesture_cooldown = 0
        self.last_gesture_time = 0
        self.zoom_smoothing = 0.7  # Smoothing factor for zoom
        self.smooth_scale = 0
        
        # Load first image
        if self.images:
            self.load_current_image()
    
    def detect_available_cameras(self):
        """Detect all available cameras"""
        available_cameras = []
        
        print("Detecting available cameras...")
        
        # Test camera indices from 0 to 10
        for i in range(11):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify camera works
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    camera_info = {
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'name': self.get_camera_name(i)
                    }
                    available_cameras.append(camera_info)
                    print(f"Camera {i}: {camera_info['name']} ({width}x{height} @ {fps}fps)")
                
                cap.release()
            
        return available_cameras
    
    def get_camera_name(self, index):
        """Get a descriptive name for the camera"""
        if index == 0:
            return "Built-in Camera (Default)"
        else:
            return f"USB Camera {index} / External Camera"
    
    def select_camera(self):
        """Allow user to select which camera to use"""
        available_cameras = self.detect_available_cameras()
        
        if not available_cameras:
            print("No cameras detected!")
            raise RuntimeError("No cameras available")
        
        if len(available_cameras) == 1:
            selected = available_cameras[0]
            print(f"Using only available camera: {selected['name']}")
            return selected['index']
        
        print(f"\nFound {len(available_cameras)} camera(s):")
        print("-" * 80)
        
        for i, cam in enumerate(available_cameras):
            print(f"[{i}] Camera {cam['index']}: {cam['name']}")
            print(f"    Resolution: {cam['width']}x{cam['height']}")
            print(f"    FPS: {cam['fps']}")
            print()
        
        while True:
            try:
                choice = input(f"Select camera (0-{len(available_cameras)-1}) or press Enter for default [0]: ").strip()
                
                if choice == "":
                    choice = 0
                else:
                    choice = int(choice)
                
                if 0 <= choice < len(available_cameras):
                    selected = available_cameras[choice]
                    print(f"Selected: {selected['name']}")
                    return selected['index']
                else:
                    print(f"Please enter a number between 0 and {len(available_cameras)-1}")
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting...")
                exit(0)
    
    def load_images(self):
        """Load all supported image files from the current directory"""
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        images = []
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                images.append(file)
        
        if not images:
            print("No image files found! Please add images to the directory.")
            return []
        
        print(f"Found {len(images)} image(s): {', '.join(images)}")
        return sorted(images)
    
    def load_current_image(self):
        """Load the current image based on index"""
        if self.images:
            img_path = self.images[self.current_image_index]
            self.current_image = cv2.imread(img_path)
            if self.current_image is None:
                print(f"Error loading image: {img_path}")
                return False
            print(f"Loaded: {img_path}")
            return True
        return False
    
    def switch_image(self, direction=1):
        """Switch to next/previous image"""
        if not self.images:
            return
        
        self.current_image_index = (self.current_image_index + direction) % len(self.images)
        self.load_current_image()
        # Reset zoom when switching images
        self.scale = 0
        self.smooth_scale = 0
        self.startDis = None
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_time > 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = current_time
    
    def draw_ui(self, img):
        """Draw UI elements on the image"""
        h, w = img.shape[:2]
        
        # Draw FPS
        if self.show_fps:
            cv2.putText(img, f"FPS: {self.current_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current camera info
        camera_text = f"Camera: {self.camera_index} - {self.get_camera_name(self.camera_index)}"
        cv2.putText(img, camera_text, (10, h - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw current image info
        if self.images:
            info_text = f"Image: {self.current_image_index + 1}/{len(self.images)} - {self.images[self.current_image_index]}"
            cv2.putText(img, info_text, (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw zoom level
        zoom_text = f"Zoom: {self.smooth_scale:+d}"
        cv2.putText(img, zoom_text, (10, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw instructions
        if self.show_instructions:
            instructions = [
                "Controls:",
                "- Pinch gesture (thumb+index) with ONE hand to zoom",
                "- Open/close thumb-index distance to zoom in/out",
                "- 'n': Next image  'p': Previous image",
                "- 'r': Reset zoom  'c': Change camera",
                "- 'i': Toggle instructions  'f': Toggle FPS",
                "- 'q': Quit"
            ]
            
            for i, instruction in enumerate(instructions):
                y_pos = 60 + i * 25
                # Semi-transparent background
                overlay = img.copy()
                cv2.rectangle(overlay, (10, y_pos - 20), (500, y_pos + 5), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                
                cv2.putText(img, instruction, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def detect_zoom_gesture(self, hands):
        """Detect and process zoom gesture using single hand"""
        if len(hands) != 1:
            self.startDis = None
            return False
        
        hand = hands[0]
        fingers = self.detector.fingersUp(hand)
        
        # Check for pinch gesture (thumb and index finger up)
        if fingers == [1, 1, 0, 0, 0]:
            # Get landmarks for thumb tip and index finger tip
            lmList = hand["lmList"]
            
            # Thumb tip (landmark 4) and Index finger tip (landmark 8)
            thumb_tip = lmList[4]
            index_tip = lmList[8]
            
            # Calculate distance between thumb and index finger
            length = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
            
            if self.startDis is None:
                self.startDis = length
                return True
            
            # Calculate scale based on finger distance change
            # Larger distance = zoom in, smaller distance = zoom out
            raw_scale = int((length - self.startDis) // 1.5)  # Adjusted sensitivity
            self.scale = max(self.min_scale, min(self.max_scale, raw_scale))
            
            # Apply smoothing
            self.smooth_scale = int(self.smooth_scale * self.zoom_smoothing + 
                                  self.scale * (1 - self.zoom_smoothing))
            
            # Update center position to hand center
            self.cx, self.cy = hand["center"]
            return True
        
        return False
    
    def draw_zoom_indicator(self, img, hands):
        """Draw visual feedback for zoom gesture"""
        if len(hands) == 1:
            hand = hands[0]
            fingers = self.detector.fingersUp(hand)
            
            # Only draw indicators when in pinch gesture
            if fingers == [1, 1, 0, 0, 0]:
                lmList = hand["lmList"]
                
                # Get thumb and index finger positions
                thumb_tip = lmList[4]
                index_tip = lmList[8]
                
                # Draw line between thumb and index finger
                cv2.line(img, (thumb_tip[0], thumb_tip[1]), 
                        (index_tip[0], index_tip[1]), (0, 255, 0), 3)
                
                # Draw circles on fingertips
                cv2.circle(img, (thumb_tip[0], thumb_tip[1]), 8, (255, 0, 0), -1)
                cv2.circle(img, (index_tip[0], index_tip[1]), 8, (255, 0, 0), -1)
                
                # Draw zoom center point
                cv2.circle(img, (self.cx, self.cy), 10, (0, 255, 0), -1)
                cv2.circle(img, (self.cx, self.cy), 15, (255, 255, 255), 2)
                
                # Display current finger distance
                distance = ((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)**0.5
                cv2.putText(img, f"Distance: {int(distance)}", 
                           (thumb_tip[0] - 50, thumb_tip[1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def change_camera(self):
        """Change to a different camera"""
        print("\n" + "="*50)
        print("CHANGING CAMERA")
        print("="*50)
        
        # Release current camera
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Select new camera
        self.camera_index = self.select_camera()
        
        # Initialize new camera
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        print(f"Successfully switched to camera {self.camera_index}")
        return True
    
    def apply_zoom_effect(self, img):
        """Apply zoom effect to the current image"""
        if self.current_image is None:
            return img
        
        try:
            h1, w1 = self.current_image.shape[:2]
            
            # Calculate new dimensions
            new_h = max(10, ((h1 + self.smooth_scale) // 2) * 2)
            new_w = max(10, ((w1 + self.smooth_scale) // 2) * 2)
            
            # Resize image
            resized_img = cv2.resize(self.current_image, (new_w, new_h))
            
            # Calculate overlay position with bounds checking
            img_h, img_w = img.shape[:2]
            
            y1 = max(0, min(img_h - new_h, self.cy - new_h // 2))
            y2 = min(img_h, y1 + new_h)
            x1 = max(0, min(img_w - new_w, self.cx - new_w // 2))
            x2 = min(img_w, x1 + new_w)
            
            # Adjust resized image if needed
            crop_y1 = max(0, (new_h // 2) - (self.cy - y1))
            crop_y2 = crop_y1 + (y2 - y1)
            crop_x1 = max(0, (new_w // 2) - (self.cx - x1))
            crop_x2 = crop_x1 + (x2 - x1)
            
            # Apply overlay
            if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                cropped_resized = resized_img[crop_y1:crop_y2, crop_x1:crop_x2]
                if cropped_resized.size > 0:
                    img[y1:y2, x1:x2] = cropped_resized
                    
        except Exception as e:
            print(f"Error applying zoom effect: {e}")
        
        return img
    
    def run(self):
        """Main application loop"""
        print("Hand Gesture Zoom Application Started!")
        print("Make sure you have image files in the current directory.")
        
        if not self.images:
            print("No images found. Please add images and restart the application.")
            return
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            # Flip image horizontally for mirror effect
            img = cv2.flip(img, 1)
            
            # Detect hands
            hands, img = self.detector.findHands(img, draw=True)
            
            # Process zoom gesture
            if self.detect_zoom_gesture(hands):
                self.draw_zoom_indicator(img, hands)
            
            # Apply zoom effect
            img = self.apply_zoom_effect(img)
            
            # Draw UI
            self.draw_ui(img)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Display image
            cv2.imshow("Hand Gesture Zoom - Enhanced", img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):  # Next image
                self.switch_image(1)
            elif key == ord('p'):  # Previous image
                self.switch_image(-1)
            elif key == ord('r'):  # Reset zoom
                self.scale = 0
                self.smooth_scale = 0
                self.startDis = None
                self.cx, self.cy = 200, 200
            elif key == ord('i'):  # Toggle instructions
                self.show_instructions = not self.show_instructions
            elif key == ord('f'):  # Toggle FPS
                self.show_fps = not self.show_fps
            elif key == ord('c'):  # Change camera
                if self.change_camera():
                    print("Camera changed successfully!")
                else:
                    print("Failed to change camera. Continuing with current camera.")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")

def main():
    """Main function to run the application"""
    print("="*60)
    print("🎥 HAND GESTURE ZOOM - ENHANCED CAMERA SELECTION")
    print("="*60)
    
    try:
        # Optional: Allow command line camera selection
        import sys
        camera_index = None
        
        if len(sys.argv) > 1:
            try:
                camera_index = int(sys.argv[1])
                print(f"Using camera index from command line: {camera_index}")
            except ValueError:
                print("Invalid camera index in command line argument")
        
        app = HandGestureZoom(camera_index)
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("- Make sure your camera is not being used by another application")
        print("- Try disconnecting and reconnecting USB cameras")
        print("- Check camera permissions in your system settings")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()