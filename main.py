import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector
import math

class IronManGestureControl:
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
        
        # Hand detector - optimized for precise finger tracking
        self.detector = HandDetector(detectionCon=0.8, maxHands=2)
        
        # Application states
        self.mode = "SELECT"  # SELECT, CONTROL
        self.selected_objects = []  # List of selected objects
        self.active_object_index = None
        
        # Selection states
        self.selection_active = False
        self.selection_start_pos = None
        self.selection_current_pos = None
        self.selection_timer = 0
        self.selection_hold_time = 1.5  # seconds to hold for selection
        
        # Gesture recognition
        self.pinch_threshold = 40  # Distance threshold for pinch gesture
        self.grab_threshold = 100  # Distance threshold for grab gesture
        self.last_pinch_distance = None
        self.baseline_distance = None
        self.gesture_active = False
        
        # Object manipulation
        self.zoom_sensitivity = 0.008
        self.move_sensitivity = 1.5
        self.rotation_sensitivity = 0.05
        self.min_zoom = 0.2
        self.max_zoom = 5.0
        
        # Smoothing
        self.smoothing_factor = 0.7
        
        # UI and visual effects
        self.show_instructions = True
        self.show_debug = True
        self.hologram_effect = True
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Colors for Iron Man theme
        self.colors = {
            'gold': (0, 215, 255),      # Tony Stark gold
            'red': (0, 0, 255),         # Iron Man red
            'blue': (255, 200, 0),      # Arc reactor blue
            'white': (255, 255, 255),
            'green': (0, 255, 0),
            'cyan': (255, 255, 0)
        }
        
        print("🤖 Iron Man Style Gesture Control initialized!")
        print("Point at objects to select them, then use gestures to manipulate!")
    
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
    
    def detect_pointing_gesture(self, hands):
        """Detect pointing gesture (index finger extended, others closed)"""
        if len(hands) != 1:
            return False, None
        
        hand = hands[0]
        fingers = self.detector.fingersUp(hand)
        lmList = hand["lmList"]
        
        # Check if only index finger is up
        if fingers[1] == 1 and sum([fingers[0], fingers[2], fingers[3], fingers[4]]) == 0:
            # Get index finger tip position
            index_tip = lmList[8]
            return True, (index_tip[0], index_tip[1])
        
        return False, None
    
    def detect_pinch_gesture(self, hands):
        """Detect pinch gesture (thumb and index finger close together)"""
        if len(hands) != 1:
            return False, 0, None
        
        hand = hands[0]
        fingers = self.detector.fingersUp(hand)
        lmList = hand["lmList"]
        
        # Check if thumb and index are up
        if fingers[0] == 1 and fingers[1] == 1:
            thumb_tip = lmList[4]
            index_tip = lmList[8]
            
            # Calculate distance
            distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + 
                               (thumb_tip[1] - index_tip[1])**2)
            
            # Calculate center point
            center = ((thumb_tip[0] + index_tip[0]) // 2, 
                     (thumb_tip[1] + index_tip[1]) // 2)
            
            is_pinch = distance < self.pinch_threshold
            return is_pinch, distance, center
        
        return False, 0, None
    
    def detect_grab_gesture(self, hands):
        """Detect grab gesture (all fingers closed into fist)"""
        if len(hands) != 1:
            return False, None
        
        hand = hands[0]
        fingers = self.detector.fingersUp(hand)
        lmList = hand["lmList"]
        
        # Check if all fingers are down (fist)
        if sum(fingers) == 0:
            # Get center of palm
            palm_center = lmList[9]  # Middle finger MCP joint
            return True, (palm_center[0], palm_center[1])
        
        return False, None
    
    def detect_two_finger_gesture(self, hands):
        """Detect two-finger gesture for zoom/rotate"""
        if len(hands) != 1:
            return False, 0, None, 0
        
        hand = hands[0]
        fingers = self.detector.fingersUp(hand)
        lmList = hand["lmList"]
        
        # Check if thumb and index are up, others down
        if fingers[0] == 1 and fingers[1] == 1 and sum(fingers[2:]) == 0:
            thumb_tip = lmList[4]
            index_tip = lmList[8]
            
            # Calculate distance and angle
            dx = index_tip[0] - thumb_tip[0]
            dy = index_tip[1] - thumb_tip[1]
            distance = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx)
            
            # Calculate center
            center = ((thumb_tip[0] + index_tip[0]) // 2, 
                     (thumb_tip[1] + index_tip[1]) // 2)
            
            return True, distance, center, angle
        
        return False, 0, None, 0
    
    def handle_object_selection(self, frame, hands):
        """Handle object selection using pointing gesture"""
        h, w = frame.shape[:2]
        
        is_pointing, point_pos = self.detect_pointing_gesture(hands)
        
        if is_pointing and point_pos:
            current_time = time.time()
            
            if not self.selection_active:
                # Start new selection
                self.selection_active = True
                self.selection_start_pos = point_pos
                self.selection_timer = current_time
                self.selection_current_pos = point_pos
            else:
                # Check if pointing at same general area
                dist_from_start = math.sqrt(
                    (point_pos[0] - self.selection_start_pos[0])**2 + 
                    (point_pos[1] - self.selection_start_pos[1])**2
                )
                
                if dist_from_start < 50:  # Within 50 pixels
                    self.selection_current_pos = point_pos
                    hold_time = current_time - self.selection_timer
                    
                    # Draw selection progress
                    progress = min(hold_time / self.selection_hold_time, 1.0)
                    
                    # Draw selection circle with progress
                    radius = int(30 + progress * 20)
                    color = self.colors['blue'] if progress < 1.0 else self.colors['green']
                    
                    cv2.circle(frame, point_pos, radius, color, 3)
                    cv2.circle(frame, point_pos, int(radius * progress), color, -1)
                    
                    # Draw progress text
                    cv2.putText(frame, f"{int(progress * 100)}%", 
                               (point_pos[0] - 20, point_pos[1] - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Complete selection when hold time reached
                    if hold_time >= self.selection_hold_time:
                        self.complete_object_selection(frame, point_pos)
                        self.selection_active = False
                else:
                    # Reset selection if moved too far
                    self.selection_active = False
        else:
            # Reset selection if not pointing
            self.selection_active = False
    
    def complete_object_selection(self, frame, center_pos):
        """Complete object selection and create bounding box"""
        h, w = frame.shape[:2]
        
        # Create selection area around the point
        selection_size = 100  # Default selection size
        
        x = max(0, center_pos[0] - selection_size // 2)
        y = max(0, center_pos[1] - selection_size // 2)
        width = min(selection_size, w - x)
        height = min(selection_size, h - y)
        
        # Extract object region
        object_region = frame[y:y+height, x:x+width].copy()
        
        # Create object data structure
        new_object = {
            'id': len(self.selected_objects),
            'bbox': (x, y, width, height),
            'original_region': object_region,
            'zoom': 1.0,
            'x_offset': 0,
            'y_offset': 0,
            'rotation': 0.0,
            'smooth_zoom': 1.0,
            'smooth_x': 0,
            'smooth_y': 0,
            'smooth_rotation': 0.0,
            'selected_time': time.time()
        }
        
        self.selected_objects.append(new_object)
        self.active_object_index = len(self.selected_objects) - 1
        self.mode = "CONTROL"
        
        print(f"✅ Object {new_object['id']} selected at {center_pos}")
        print("Now use gestures to manipulate the object!")
    
    def handle_object_manipulation(self, frame, hands):
        """Handle object manipulation using various gestures"""
        if not self.selected_objects or self.active_object_index is None:
            return
        
        if self.active_object_index >= len(self.selected_objects):
            return
        
        active_obj = self.selected_objects[self.active_object_index]
        
        # Detect two-finger gesture for zoom and rotate
        is_two_finger, distance, center, angle = self.detect_two_finger_gesture(hands)
        
        if is_two_finger and center:
            if self.baseline_distance is None:
                self.baseline_distance = distance
                self.last_pinch_distance = distance
                self.gesture_active = True
            else:
                # Calculate zoom change
                distance_change = distance - self.baseline_distance
                zoom_change = distance_change * self.zoom_sensitivity
                
                # Update zoom
                new_zoom = active_obj['zoom'] + zoom_change
                active_obj['zoom'] = np.clip(new_zoom, self.min_zoom, self.max_zoom)
                
                # Update position based on gesture center
                if hasattr(self, 'last_gesture_center'):
                    dx = (center[0] - self.last_gesture_center[0]) * self.move_sensitivity
                    dy = (center[1] - self.last_gesture_center[1]) * self.move_sensitivity
                    active_obj['x_offset'] += dx
                    active_obj['y_offset'] += dy
                
                self.last_gesture_center = center
                self.baseline_distance = distance
        else:
            # Check for grab gesture (move only)
            is_grab, grab_center = self.detect_grab_gesture(hands)
            
            if is_grab and grab_center:
                if hasattr(self, 'last_grab_center'):
                    dx = (grab_center[0] - self.last_grab_center[0]) * self.move_sensitivity
                    dy = (grab_center[1] - self.last_grab_center[1]) * self.move_sensitivity
                    active_obj['x_offset'] += dx
                    active_obj['y_offset'] += dy
                
                self.last_grab_center = grab_center
            else:
                # Reset gesture tracking
                self.baseline_distance = None
                self.gesture_active = False
                if hasattr(self, 'last_gesture_center'):
                    delattr(self, 'last_gesture_center')
                if hasattr(self, 'last_grab_center'):
                    delattr(self, 'last_grab_center')
    
    def render_objects(self, frame):
        """Render all selected objects with transformations"""
        h, w = frame.shape[:2]
        
        for i, obj in enumerate(self.selected_objects):
            # Apply smoothing
            smoothing = self.smoothing_factor
            obj['smooth_zoom'] = obj['smooth_zoom'] * smoothing + obj['zoom'] * (1 - smoothing)
            obj['smooth_x'] = obj['smooth_x'] * smoothing + obj['x_offset'] * (1 - smoothing)
            obj['smooth_y'] = obj['smooth_y'] * smoothing + obj['y_offset'] * (1 - smoothing)
            
            # Get original bbox and region
            x, y, width, height = obj['bbox']
            original_region = obj['original_region']
            
            # Calculate new dimensions
            new_width = max(10, int(width * obj['smooth_zoom']))
            new_height = max(10, int(height * obj['smooth_zoom']))
            
            # Resize object
            if new_width > 0 and new_height > 0:
                resized_object = cv2.resize(original_region, (new_width, new_height), 
                                          interpolation=cv2.INTER_LINEAR)
            else:
                resized_object = original_region
            
            # Calculate placement position
            center_x = x + width // 2 + int(obj['smooth_x'])
            center_y = y + height // 2 + int(obj['smooth_y'])
            
            place_x = center_x - new_width // 2
            place_y = center_y - new_height // 2
            
            # Ensure placement is within bounds
            place_x = max(0, min(place_x, w - new_width))
            place_y = max(0, min(place_y, h - new_height))
            
            # Place object on frame
            if place_x + new_width <= w and place_y + new_height <= h:
                if self.hologram_effect:
                    # Apply hologram effect
                    hologram_object = self.apply_hologram_effect(resized_object)
                    frame[place_y:place_y+new_height, place_x:place_x+new_width] = hologram_object
                else:
                    frame[place_y:place_y+new_height, place_x:place_x+new_width] = resized_object
                
                # Draw border
                border_color = self.colors['gold'] if i == self.active_object_index else self.colors['blue']
                thickness = 3 if i == self.active_object_index else 2
                cv2.rectangle(frame, (place_x, place_y), 
                             (place_x + new_width, place_y + new_height), border_color, thickness)
                
                # Draw object ID
                cv2.putText(frame, f"OBJ-{obj['id']}", 
                           (place_x, place_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, border_color, 2)
            
            # Draw original location with dashed line
            self.draw_dashed_rectangle(frame, (x, y, width, height), self.colors['red'], 1)
    
    def apply_hologram_effect(self, img):
        """Apply hologram-like visual effect"""
        # Create a slight blue tint
        hologram = img.copy().astype(np.float32)
        
        # Add blue tint
        hologram[:, :, 0] = np.clip(hologram[:, :, 0] * 1.2, 0, 255)  # Blue channel
        hologram[:, :, 1] = np.clip(hologram[:, :, 1] * 0.8, 0, 255)  # Green channel
        hologram[:, :, 2] = np.clip(hologram[:, :, 2] * 0.6, 0, 255)  # Red channel
        
        # Add some transparency effect
        alpha = 0.85
        hologram = hologram * alpha
        
        return hologram.astype(np.uint8)
    
    def draw_dashed_rectangle(self, img, bbox, color, thickness):
        """Draw dashed rectangle"""
        x, y, w, h = bbox
        dash_length = 10
        
        # Top line
        for i in range(0, w, dash_length * 2):
            start_x = x + i
            end_x = min(x + i + dash_length, x + w)
            cv2.line(img, (start_x, y), (end_x, y), color, thickness)
        
        # Bottom line
        for i in range(0, w, dash_length * 2):
            start_x = x + i
            end_x = min(x + i + dash_length, x + w)
            cv2.line(img, (start_x, y + h), (end_x, y + h), color, thickness)
        
        # Left line
        for i in range(0, h, dash_length * 2):
            start_y = y + i
            end_y = min(y + i + dash_length, y + h)
            cv2.line(img, (x, start_y), (x, end_y), color, thickness)
        
        # Right line
        for i in range(0, h, dash_length * 2):
            start_y = y + i
            end_y = min(y + i + dash_length, y + h)
            cv2.line(img, (x + w, start_y), (x + w, end_y), color, thickness)
    
    def draw_gesture_indicators(self, img, hands):
        """Draw visual indicators for current gestures"""
        if not hands:
            return
        
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = self.detector.fingersUp(hand)
        
        # Draw hand landmarks with Iron Man styling
        if self.show_debug:
            for i, lm in enumerate(lmList):
                cv2.circle(img, (lm[0], lm[1]), 5, self.colors['gold'], -1)
        
        # Check for pointing gesture
        is_pointing, point_pos = self.detect_pointing_gesture(hands)
        if is_pointing:
            cv2.circle(img, point_pos, 15, self.colors['red'], 3)
            cv2.putText(img, "POINTING", (point_pos[0] - 40, point_pos[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['red'], 2)
        
        # Check for two-finger gesture
        is_two_finger, distance, center, angle = self.detect_two_finger_gesture(hands)
        if is_two_finger:
            thumb_tip = lmList[4]
            index_tip = lmList[8]
            
            # Draw line between fingers
            cv2.line(img, (thumb_tip[0], thumb_tip[1]), 
                    (index_tip[0], index_tip[1]), self.colors['cyan'], 4)
            
            # Draw center point
            cv2.circle(img, center, 8, self.colors['green'], -1)
            
            # Display distance
            cv2.putText(img, f"DIST: {int(distance)}", 
                       (center[0] - 50, center[1] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['cyan'], 2)
        
        # Check for grab gesture
        is_grab, grab_center = self.detect_grab_gesture(hands)
        if is_grab:
            cv2.circle(img, grab_center, 20, self.colors['red'], 4)
            cv2.putText(img, "GRAB", (grab_center[0] - 25, grab_center[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['red'], 2)
    
    def draw_ui(self, img):
        """Draw Iron Man styled user interface"""
        h, w = img.shape[:2]
        
        # Mode indicator with Iron Man styling
        mode_color = self.colors['green'] if self.mode == "CONTROL" else self.colors['gold']
        cv2.putText(img, f"◉ MODE: {self.mode}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 2)
        
        # FPS with arc reactor styling
        cv2.circle(img, (80, 70), 25, self.colors['blue'], 2)
        cv2.putText(img, f"{self.current_fps}", (70, 78), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['blue'], 2)
        
        # Object count
        cv2.putText(img, f"OBJECTS: {len(self.selected_objects)}", (10, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['white'], 2)
        
        if self.active_object_index is not None and self.selected_objects:
            active_obj = self.selected_objects[self.active_object_index]
            cv2.putText(img, f"ACTIVE: OBJ-{active_obj['id']}", (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['gold'], 2)
            
            # Object properties
            zoom_text = f"ZOOM: {active_obj['smooth_zoom']:.2f}x"
            cv2.putText(img, zoom_text, (200, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
            
            pos_text = f"POS: ({int(active_obj['smooth_x'])}, {int(active_obj['smooth_y'])})"
            cv2.putText(img, pos_text, (200, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        
        # Instructions overlay
        if self.show_instructions:
            self.draw_instructions_overlay(img)
    
    def draw_instructions_overlay(self, img):
        """Draw Iron Man styled instructions overlay"""
        h, w = img.shape[:2]
        
        # Create semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (w - 450, 10), (w - 10, 350), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        # Add border
        cv2.rectangle(img, (w - 450, 10), (w - 10, 350), self.colors['gold'], 2)
        
        instructions = [
            "🤖 IRON MAN GESTURE CONTROL",
            "",
            "SELECTION MODE:",
            "👆 Point at object (hold 1.5s)",
            "   Keep index finger extended",
            "",
            "CONTROL MODE:",
            "👌 Thumb + Index = Zoom & Move",
            "   Spread apart = Zoom In",
            "   Close together = Zoom Out",
            "   Move hand = Move Object",
            "",
            "✊ Fist = Grab & Move Only",
            "",
            "CONTROLS:",
            "'s': New Selection  'r': Reset",
            "'n': Next Object   'h': Hologram",
            "'i': Instructions  'q': Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = 35 + i * 18
            
            if instruction.startswith('🤖'):
                color = self.colors['gold']
                font_scale = 0.5
            elif instruction.startswith(('👆', '👌', '✊')):
                color = self.colors['cyan']
                font_scale = 0.45
            elif instruction.startswith(("'s'", "'n'", "'i'")):
                color = self.colors['blue']
                font_scale = 0.4
            else:
                color = self.colors['white']
                font_scale = 0.45
            
            cv2.putText(img, instruction, (w - 440, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    
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
        print("🤖 IRON MAN STYLE GESTURE CONTROL - Tony Stark Mode Activated!")
        print("="*70)
        print("1. Point at objects to select them (hold for 1.5 seconds)")
        print("2. Use thumb + index finger gestures to zoom and move")
        print("3. Make a fist to grab and move objects")
        print("4. Just like Tony Stark in Iron Man 2! 🦾")
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Camera read failed")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            hands, frame = self.detector.findHands(frame, draw=False)
            
            # Process based on current mode
            if self.mode == "SELECT":
                self.handle_object_selection(frame, hands)
            elif self.mode == "CONTROL":
                self.handle_object_manipulation(frame, hands)
                self.render_objects(frame)
            
            # Draw gesture indicators
            self.draw_gesture_indicators(frame, hands)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Calculate FPS
            self.calculate_fps()
            
            # Display frame
            cv2.imshow("Iron Man Gesture Control", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):  # Switch to selection mode
                self.mode = "SELECT"
                self.active_object_index = None
                print("🎯 Selection mode activated - point at objects to select!")
            elif key == ord('r'):  # Reset active object
                if self.active_object_index is not None and self.selected_objects:
                    obj = self.selected_objects[self.active_object_index]
                    obj['zoom'] = 1.0
                    obj['x_offset'] = 0
                    obj['y_offset'] = 0
                    obj['rotation'] = 0.0
                    print(f"🔄 Object {obj['id']} reset!")
            elif key == ord('n'):  # Next object
                if self.selected_objects:
                    self.active_object_index = (self.active_object_index + 1) % len(self.selected_objects)
                    active_obj = self.selected_objects[self.active_object_index]
                    print(f"📋 Switched to Object {active_obj['id']}")
            elif key == ord('h'):  # Toggle hologram effect
                self.hologram_effect = not self.hologram_effect
                effect_status = "ON" if self.hologram_effect else "OFF"
                print(f"🌟 Hologram effect: {effect_status}")
            elif key == ord('i'):  # Toggle instructions
                self.show_instructions = not self.show_instructions
            elif key == ord('d'):  # Toggle debug
                self.show_debug = not self.show_debug
            elif key == ord('c'):  # Clear all objects
                self.selected_objects.clear()
                self.active_object_index = None
                self.mode = "SELECT"
                print("🗑️ All objects cleared!")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("🤖 Iron Man Gesture Control deactivated!")

def main():
    """Main function"""
    print("🚀 Initializing Iron Man Gesture Control System...")
    print("🦾 Channeling Tony Stark's genius...")
    
    try:
        import sys
        camera_index = None
        
        if len(sys.argv) > 1:
            try:
                camera_index = int(sys.argv[1])
            except ValueError:
                print("Invalid camera index")
        
        app = IronManGestureControl(camera_index)
        app.run()
        
    except KeyboardInterrupt:
        print("\n👋 System shutdown initiated by user")
    except Exception as e:
        print(f"❌ System Error: {e}")
        print("\n🔧 Required Dependencies:")
        print("pip install opencv-python cvzone numpy")
        print("\n📋 Troubleshooting:")
        print("- Ensure your camera is connected and working")
        print("- Check that cvzone is properly installed")
        print("- Make sure no other applications are using the camera")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()