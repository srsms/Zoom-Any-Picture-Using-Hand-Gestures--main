import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

class ObjectSelectionGestureControl:
    def __init__(self, camera_index=None):
        # Camera setup
        self.camera_index = self.select_camera() if camera_index is None else camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {self.camera_index} not accessible")
        
        # Hand detector - optimized for thumb and index detection
        self.detector = HandDetector(detectionCon=0.8, maxHands=1)
        
        # Application states
        self.mode = "SELECT"  # SELECT, CONTROL
        self.selected_object = None
        self.object_bbox = None
        self.selection_points = []
        
        # Object tracking and manipulation
        self.object_zoom = 1.0
        self.object_x = 0
        self.object_y = 0
        self.min_zoom = 0.3
        self.max_zoom = 5.0
        
        # Gesture detection for thumb-index only
        self.baseline_distance = None
        self.last_thumb_index_distance = None
        self.gesture_active = False
        self.gesture_threshold = 30  # Minimum distance to consider gesture
        self.zoom_sensitivity = 0.005
        self.move_sensitivity = 2.0
        
        # Smoothing
        self.smooth_zoom = 1.0
        self.smooth_x = 0
        self.smooth_y = 0
        self.smoothing_factor = 0.7
        
        # UI elements
        self.show_instructions = True
        self.show_debug = False
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Mouse callback for object selection
        self.mouse_selecting = False
        self.selection_start = None
        self.selection_end = None
        
        print("Object Selection & Gesture Control initialized!")
        print("First select an object, then use thumb-index gestures to control it!")
    
    def detect_available_cameras(self):
        """Detect all available cameras"""
        available_cameras = []
        print("Detecting cameras...")
        
        for i in range(8):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    camera_info = {
                        'index': i,
                        'width': width,
                        'height': height,
                        'name': f"Camera {i}" if i > 0 else "Built-in Camera"
                    }
                    available_cameras.append(camera_info)
                    print(f"✓ Camera {i}: {width}x{height}")
                
                cap.release()
            
        return available_cameras
    
    def select_camera(self):
        """Camera selection with user input"""
        cameras = self.detect_available_cameras()
        
        if not cameras:
            raise RuntimeError("No cameras found!")
        
        if len(cameras) == 1:
            print(f"Using: {cameras[0]['name']}")
            return cameras[0]['index']
        
        print(f"\nAvailable cameras:")
        for i, cam in enumerate(cameras):
            print(f"[{i}] {cam['name']} - {cam['width']}x{cam['height']}")
        
        while True:
            try:
                choice = input(f"Select camera (0-{len(cameras)-1}) [0]: ").strip()
                choice = 0 if choice == "" else int(choice)
                
                if 0 <= choice < len(cameras):
                    print(f"Selected: {cameras[choice]['name']}")
                    return cameras[choice]['index']
                else:
                    print(f"Enter 0-{len(cameras)-1}")
                    
            except (ValueError, KeyboardInterrupt):
                return cameras[0]['index']
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for object selection"""
        if self.mode != "SELECT":
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_selecting = True
            self.selection_start = (x, y)
            self.selection_end = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_selecting:
            self.selection_end = (x, y)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_selecting = False
            if self.selection_start and self.selection_end:
                # Create bounding box
                x1, y1 = self.selection_start
                x2, y2 = self.selection_end
                
                # Ensure proper ordering
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Minimum selection size
                if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:
                    self.object_bbox = (x1, y1, x2 - x1, y2 - y1)
                    self.mode = "CONTROL"
                    self.reset_object_transform()
                    print(f"✓ Object selected! Bbox: {self.object_bbox}")
                    print("Now use thumb-index gestures to control the object!")
                else:
                    print("Selection too small - try again")
                
                self.selection_start = None
                self.selection_end = None
    
    def reset_object_transform(self):
        """Reset object transformation parameters"""
        self.object_zoom = 1.0
        self.object_x = 0
        self.object_y = 0
        self.smooth_zoom = 1.0
        self.smooth_x = 0
        self.smooth_y = 0
        self.baseline_distance = None
        self.last_thumb_index_distance = None
    
    def detect_thumb_index_gesture(self, hands):
        """Detect and process thumb-index finger gestures"""
        if len(hands) != 1:
            self.gesture_active = False
            self.baseline_distance = None
            return False
        
        hand = hands[0]
        fingers = self.detector.fingersUp(hand)
        lmList = hand["lmList"]
        
        # Check if only thumb and index finger are up
        thumb_up = fingers[0] == 1
        index_up = fingers[1] == 1
        other_fingers_down = sum(fingers[2:]) == 0
        
        if thumb_up and index_up and other_fingers_down:
            # Get thumb tip and index finger tip positions
            thumb_tip = lmList[4]  # Thumb tip
            index_tip = lmList[8]  # Index finger tip
            
            # Calculate distance between thumb and index finger
            distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + 
                             (thumb_tip[1] - index_tip[1])**2)
            
            # Initialize baseline distance
            if self.baseline_distance is None:
                self.baseline_distance = distance
                self.last_thumb_index_distance = distance
                self.gesture_active = True
                return True
            
            # Only process if distance is above threshold (fingers not too close)
            if distance > self.gesture_threshold:
                # Calculate zoom based on distance change from baseline
                distance_change = distance - self.baseline_distance
                zoom_change = distance_change * self.zoom_sensitivity
                
                # Update zoom level
                new_zoom = self.object_zoom + zoom_change
                self.object_zoom = np.clip(new_zoom, self.min_zoom, self.max_zoom)
                
                # Calculate movement based on midpoint between thumb and index
                midpoint_x = (thumb_tip[0] + index_tip[0]) // 2
                midpoint_y = (thumb_tip[1] + index_tip[1]) // 2
                
                # Move object based on gesture center movement
                if hasattr(self, 'last_midpoint'):
                    dx = (midpoint_x - self.last_midpoint[0]) * self.move_sensitivity
                    dy = (midpoint_y - self.last_midpoint[1]) * self.move_sensitivity
                    self.object_x += dx
                    self.object_y += dy
                
                self.last_midpoint = (midpoint_x, midpoint_y)
                self.last_thumb_index_distance = distance
                self.gesture_active = True
                
                return True
            
        self.gesture_active = False
        if not thumb_up or not index_up:
            self.baseline_distance = None
            
        return False
    
    def extract_and_transform_object(self, frame):
        """Extract selected object and apply transformations"""
        if self.object_bbox is None:
            return frame
        
        h, w = frame.shape[:2]
        x, y, w_bbox, h_bbox = self.object_bbox
        
        # Ensure bbox is within frame bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        w_bbox = min(w_bbox, w - x)
        h_bbox = min(h_bbox, h - y)
        
        if w_bbox <= 0 or h_bbox <= 0:
            return frame
        
        # Extract object region
        object_region = frame[y:y+h_bbox, x:x+w_bbox].copy()
        
        # Apply smooth transformations
        self.smooth_zoom = self.smooth_zoom * self.smoothing_factor + self.object_zoom * (1 - self.smoothing_factor)
        self.smooth_x = self.smooth_x * self.smoothing_factor + self.object_x * (1 - self.smoothing_factor)
        self.smooth_y = self.smooth_y * self.smoothing_factor + self.object_y * (1 - self.smoothing_factor)
        
        # Calculate new dimensions
        new_width = max(10, int(w_bbox * self.smooth_zoom))
        new_height = max(10, int(h_bbox * self.smooth_zoom))
        
        # Resize object
        if new_width > 0 and new_height > 0:
            transformed_object = cv2.resize(object_region, (new_width, new_height), 
                                          interpolation=cv2.INTER_LINEAR)
        else:
            transformed_object = object_region
        
        # Create output frame
        result_frame = frame.copy()
        
        # Calculate placement position
        center_x = x + w_bbox // 2 + int(self.smooth_x)
        center_y = y + h_bbox // 2 + int(self.smooth_y)
        
        # Calculate top-left corner for placement
        place_x = center_x - new_width // 2
        place_y = center_y - new_height // 2
        
        # Ensure placement is within bounds
        place_x = max(0, min(place_x, w - new_width))
        place_y = max(0, min(place_y, h - new_height))
        
        # Place transformed object
        if place_x + new_width <= w and place_y + new_height <= h:
            # Create mask for blending
            mask = np.ones((new_height, new_width, 3), dtype=np.float32)
            
            # Blend the transformed object onto the frame
            result_frame[place_y:place_y+new_height, place_x:place_x+new_width] = transformed_object
            
            # Draw border around transformed object
            cv2.rectangle(result_frame, (place_x, place_y), 
                         (place_x + new_width, place_y + new_height), (0, 255, 0), 2)
        
        return result_frame
    
    def draw_gesture_indicators(self, img, hands):
        """Draw visual indicators for thumb-index gestures"""
        if len(hands) != 1:
            return
        
        hand = hands[0]
        fingers = self.detector.fingersUp(hand)
        lmList = hand["lmList"]
        
        # Check for thumb-index gesture
        if fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0:
            thumb_tip = lmList[4]
            index_tip = lmList[8]
            
            # Draw line between thumb and index
            cv2.line(img, thumb_tip[:2], index_tip[:2], (0, 255, 255), 4)
            
            # Draw circles on fingertips
            cv2.circle(img, thumb_tip[:2], 12, (255, 0, 0), -1)
            cv2.circle(img, index_tip[:2], 12, (255, 0, 0), -1)
            
            # Draw midpoint
            midpoint = ((thumb_tip[0] + index_tip[0]) // 2, 
                       (thumb_tip[1] + index_tip[1]) // 2)
            cv2.circle(img, midpoint, 8, (0, 255, 0), -1)
            
            # Display distance
            distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + 
                             (thumb_tip[1] - index_tip[1])**2)
            
            cv2.putText(img, f"Distance: {int(distance)}", 
                       (thumb_tip[0] - 60, thumb_tip[1] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.gesture_active:
                cv2.putText(img, "GESTURE ACTIVE", 
                           (midpoint[0] - 80, midpoint[1] + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def draw_ui(self, img):
        """Draw user interface elements"""
        h, w = img.shape[:2]
        
        # Mode indicator
        mode_color = (0, 255, 0) if self.mode == "CONTROL" else (0, 255, 255)
        cv2.putText(img, f"MODE: {self.mode}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 2)
        
        # FPS
        cv2.putText(img, f"FPS: {self.current_fps}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.mode == "SELECT":
            # Selection instructions
            instructions = [
                "OBJECT SELECTION MODE",
                "1. Click and drag to select an object",
                "2. Make sure to select the entire object",
                "3. Release mouse to confirm selection"
            ]
            
            for i, instruction in enumerate(instructions):
                y_pos = h - 120 + i * 30
                cv2.putText(img, instruction, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw current selection rectangle
            if self.mouse_selecting and self.selection_start and self.selection_end:
                cv2.rectangle(img, self.selection_start, self.selection_end, (0, 255, 255), 2)
                
        elif self.mode == "CONTROL":
            # Control instructions and info
            cv2.putText(img, f"Zoom: {self.smooth_zoom:.2f}x", (10, h - 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(img, f"Position: ({int(self.smooth_x)}, {int(self.smooth_y)})", 
                       (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw original object boundary
            if self.object_bbox:
                x, y, w_bbox, h_bbox = self.object_bbox
                cv2.rectangle(img, (x, y), (x + w_bbox, y + h_bbox), (255, 0, 0), 2)
                cv2.putText(img, "ORIGINAL", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Instructions
        if self.show_instructions:
            inst_bg = np.zeros((200, 400, 3), dtype=np.uint8)
            inst_bg[:] = (0, 0, 0)
            
            instructions = [
                "THUMB-INDEX GESTURE CONTROL:",
                "",
                "👆 Hold THUMB + INDEX up only",
                "   (keep other fingers down)",
                "",
                "🔍 Spread fingers = ZOOM IN",
                "🔍 Close fingers = ZOOM OUT", 
                "👋 Move hand = MOVE OBJECT",
                "",
                "CONTROLS:",
                "'r': Reset  's': New selection",
                "'i': Instructions  'q': Quit"
            ]
            
            overlay = img.copy()
            cv2.rectangle(overlay, (w - 420, 10), (w - 10, 250), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
            
            for i, instruction in enumerate(instructions):
                y_pos = 35 + i * 18
                color = (0, 255, 255) if instruction.startswith(('👆', '🔍', '👋')) else (255, 255, 255)
                cv2.putText(img, instruction, (w - 410, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    def calculate_fps(self):
        """Calculate FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_time > 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = current_time
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*70)
        print("🎯 OBJECT SELECTION & THUMB-INDEX GESTURE CONTROL")
        print("="*70)
        print("1. First SELECT an object by clicking and dragging")
        print("2. Then use THUMB + INDEX gestures to control it")
        print("3. Keep other fingers DOWN for gesture recognition")
        
        # Set up mouse callback
        cv2.namedWindow("Object Selection & Gesture Control")
        cv2.setMouseCallback("Object Selection & Gesture Control", self.mouse_callback)
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Camera read failed")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            hands, frame = self.detector.findHands(frame, draw=True)
            
            # Process based on current mode
            if self.mode == "CONTROL" and self.object_bbox is not None:
                # Detect thumb-index gestures
                self.detect_thumb_index_gesture(hands)
                
                # Apply object transformation
                frame = self.extract_and_transform_object(frame)
                
                # Draw gesture indicators
                self.draw_gesture_indicators(frame, hands)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Display frame
            cv2.imshow("Object Selection & Gesture Control", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):  # New selection
                self.mode = "SELECT"
                self.object_bbox = None
                self.reset_object_transform()
                print("Selection mode activated - click and drag to select object")
            elif key == ord('r'):  # Reset transformation
                if self.mode == "CONTROL":
                    self.reset_object_transform()
                    print("Object transformation reset")
            elif key == ord('i'):  # Toggle instructions
                self.show_instructions = not self.show_instructions
            elif key == ord('d'):  # Toggle debug
                self.show_debug = not self.show_debug
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed!")

def main():
    """Main function"""
    print("🚀 Starting Object Selection & Gesture Control...")
    
    try:
        import sys
        camera_index = None
        
        if len(sys.argv) > 1:
            try:
                camera_index = int(sys.argv[1])
            except ValueError:
                print("Invalid camera index")
        
        app = ObjectSelectionGestureControl(camera_index)
        app.run()
        
    except KeyboardInterrupt:
        print("\n👋 Application interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Requirements:")
        print("pip install opencv-python cvzone")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()