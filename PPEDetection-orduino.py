import pyfirmata
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import pyttsx3
import pandas as pd
from datetime import datetime
import os
import numpy as np
import pickle

# Arduino setup
port = "COM5"
try:
    board = pyfirmata.Arduino(port)
    # LED setup
    green_led = board.get_pin("d:13:o")
    red_led = board.get_pin("d:12:o")
    
    # Stepper motor control pins (4-wire)
    stepper_pins = [8, 9, 10, 11]
    steps_sequence = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1]
    ]
    for pin in stepper_pins:
        board.digital[pin].mode = pyfirmata.OUTPUT
    
    arduino_available = True
    print("Arduino connected successfully.")
except Exception as e:
    print(f"Arduino not connected: {e}. Running in simulation mode.")
    arduino_available = False

def rotate_stepper(steps=100, delay=0.01):
    if not arduino_available:
        print(f"[SIMULATION] Rotating stepper motor {steps} steps")
        return
    
    for _ in range(steps):
        for step in steps_sequence:
            for pin, val in zip(stepper_pins, step):
                board.digital[pin].write(val)
            time.sleep(delay)

def set_led(green_on, red_on):
    if not arduino_available:
        status = "GREEN" if green_on else "RED" if red_on else "OFF"
        print(f"[SIMULATION] LED: {status}")
        return
    
    try:
        if green_led:
            green_led.write(1 if green_on else 0)
        if red_led:
            red_led.write(1 if red_on else 0)
    except:
        print("LED control failed")

# Load YOLO model
try:
    model = YOLO("ppe.pt")
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Using dummy detection for testing.")
    model = None

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

# Face recognition setup using OpenCV
class SimpleFacerec:
    def __init__(self):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.labels = {}
        self.model_file = "face_recognizer_model.yml"
        self.labels_file = "face_labels.pkl"
        self.frame_resizing = 0.5  # For faster processing

    def load_encoding_images(self, images_path):
        """Load and train face recognition model from images folder"""
        print(f"Loading images from: {images_path}")
        
        if not os.path.exists(images_path):
            print(f"Warning: Images folder '{images_path}' not found.")
            os.makedirs(images_path, exist_ok=True)
            print(f"Created empty directory: {images_path}")
            return False
        
        faces = []
        labels = []
        label_id = 0
        
        # Get all subdirectories (each representing a person)
        if not os.listdir(images_path):
            print("No person directories found in Images folder.")
            print("Please create folders named with person names and add their photos inside.")
            return False
        
        for person_name in os.listdir(images_path):
            person_path = os.path.join(images_path, person_name)
            
            if os.path.isdir(person_path):
                print(f"Processing: {person_name}")
                self.labels[label_id] = person_name
                
                # Get all images for this person
                image_files = [f for f in os.listdir(person_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not image_files:
                    print(f"  No images found for {person_name}")
                    continue
                
                for image_file in image_files:
                    img_path = os.path.join(person_path, image_file)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces_detected = self.face_cascade.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        
                        if len(faces_detected) == 0:
                            print(f"  No face found in {image_file}")
                            continue
                        
                        for (x, y, w, h) in faces_detected:
                            face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                            faces.append(face_roi)
                            labels.append(label_id)
                            print(f"  ✓ Added face from {image_file}")
                label_id += 1
        
        if not faces:
            print("No faces found for training.")
            return False
        
        # Train the model
        print(f"Training with {len(faces)} face samples from {len(self.labels)} persons...")
        self.face_recognizer.train(faces, np.array(labels))
        self.face_recognizer.save(self.model_file)
        
        # Save labels
        with open(self.labels_file, 'wb') as f:
            pickle.dump(self.labels, f)
        
        print(f"Model trained and saved to {self.model_file}")
        print(f"Labels saved to {self.labels_file}")
        return True

    def load_trained_model(self):
        """Load previously trained model"""
        if os.path.exists(self.model_file) and os.path.exists(self.labels_file):
            try:
                self.face_recognizer.read(self.model_file)
                with open(self.labels_file, 'rb') as f:
                    self.labels = pickle.load(f)
                print(f"Loaded trained model with {len(self.labels)} persons")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        return False

    def detect_known_faces(self, frame):
        """Detect and recognize faces in the frame"""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces_detected = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_locations = []
        face_names = []
        
        for (x, y, w, h) in faces_detected:
            face_roi = gray[y:y+h, x:x+w]
            
            # Predict the label
            label, confidence = self.face_recognizer.predict(face_roi)
            
            # Adjust confidence threshold (lower is better for LBPH)
            confidence_threshold = 70
            
            if confidence < confidence_threshold and label in self.labels:
                name = self.labels[label]
            else:
                name = "Unknown"
            
            # Scale coordinates back to original frame size
            x_orig = int(x / self.frame_resizing)
            y_orig = int(y / self.frame_resizing)
            w_orig = int(w / self.frame_resizing)
            h_orig = int(h / self.frame_resizing)
            
            face_locations.append((y_orig, y_orig + h_orig, x_orig + w_orig, x_orig))
            face_names.append(name)
        
        return face_locations, face_names

# Initialize face recognizer
sfr = SimpleFacerec()

# Try to load existing model, otherwise train new one
if not sfr.load_trained_model():
    print("No trained model found. Starting training...")
    if not sfr.load_encoding_images("Images/"):
        print("Could not train model. Please add person folders with photos to Images/ directory.")
        print("Example: Images/John_Doe/photo1.jpg, Images/John_Doe/photo2.jpg")

# Text-to-speech setup
engine = pyttsx3.init()

# Configure speech engine
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume level

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Log setup
log_file = "recognition_log.xlsx"

def speak(text):
    """Speak text using text-to-speech"""
    print(f"Speaking: {text}")
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")

def update_log(name):
    """Update the recognition log"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {"Name": name, "Time": current_time}
    
    try:
        if os.path.exists(log_file):
            df = pd.read_excel(log_file)
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([new_entry])
        
        df.to_excel(log_file, index=False)
        return df
    except Exception as e:
        print(f"Error updating log: {e}")
        return pd.DataFrame(columns=["Name", "Time"])

# State management
last_access_granted = None
last_denied_message = None
access_cooldown = 5  # seconds

# Initial prompt
speak("Please enter first person")
rotate_stepper(steps=100)

print("\n" + "="*50)
print("PPE Detection System Started")
print("Press 'q' to quit")
print("="*50)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    
    # Create display image
    display_img = img.copy()
    
    # Detect faces
    face_locations, face_names = sfr.detect_known_faces(img)
    
    # Check for multiple faces
    if len(face_locations) > 1:
        cv2.putText(display_img, "MULTIPLE PERSONS DETECTED", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        current_time = time.time()
        if last_denied_message is None or (current_time - last_denied_message) > access_cooldown:
            speak("Only one person allowed")
            last_denied_message = current_time
        
        set_led(False, True)
        cv2.imshow("PPE Detection System", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Process if face is detected
    current_name = "Unknown"
    if len(face_locations) == 1:
        current_name = face_names[0]
        print(f"Detected: {current_name}")
        
        # Draw face rectangle and name
        y1, y2, x2, x1 = face_locations[0]
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(display_img, current_name, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
        
        # Update log
        update_log(current_name)
    
    # Run PPE detection
    ppe_ok = {"Hardhat": False, "Mask": False, "Safety Vest": False}
    
    if model:
        try:
            results = model(img, stream=True)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]
                    
                    if conf > 0.5:
                        color = (0, 255, 0) if 'NO-' not in currentClass else (0, 0, 255)
                        
                        # Draw detection
                        cvzone.putTextRect(display_img, f'{currentClass} {conf}', (x1, y1 - 10),
                                         scale=1, thickness=2, colorR=color, 
                                         colorT=(255, 255, 255), colorB=color, offset=5)
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 3)
                        
                        # Update PPE status
                        base_class = currentClass.replace('NO-', '')
                        if base_class in ppe_ok:
                            ppe_ok[base_class] = ('NO-' not in currentClass)
        
        except Exception as e:
            print(f"YOLO detection error: {e}")
    
    # Display information panel
    panel_y = 30
    cv2.putText(display_img, f"Person: {current_name}", (10, panel_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Display PPE status
    panel_y += 40
    for i, (item, is_ok) in enumerate(ppe_ok.items()):
        color = (0, 255, 0) if is_ok else (0, 0, 255)
        status = "OK" if is_ok else "MISSING"
        cv2.putText(display_img, f"{item}: {status}", (10, panel_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        panel_y += 30
    
    # Check access conditions
    all_ppe_ok = all(ppe_ok.values())
    current_time = time.time()
    
    if all_ppe_ok and current_name != "Unknown":
        # ACCESS GRANTED
        access_status = "ACCESS GRANTED"
        status_color = (0, 255, 0)
        set_led(True, False)
        
        if last_access_granted is None or (current_time - last_access_granted) > access_cooldown:
            speak(f"Access granted for {current_name}")
            rotate_stepper(steps=100)
            last_access_granted = current_time
            last_denied_message = None  # Reset denial cooldown
        
    else:
        # ACCESS DENIED
        access_status = "ACCESS DENIED"
        status_color = (0, 0, 255)
        set_led(False, True)
        
        if last_denied_message is None or (current_time - last_denied_message) > access_cooldown:
            reasons = []
            if current_name == "Unknown":
                reasons.append("Unknown person")
            for item, is_ok in ppe_ok.items():
                if not is_ok:
                    reasons.append(f"Missing {item}")
            
            if reasons:
                denial_msg = "Access denied. " + ", ".join(reasons)
                speak(denial_msg)
                last_denied_message = current_time
    
    # Display access status
    cv2.putText(display_img, access_status, (display_img.shape[1] - 300, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(display_img, timestamp, (10, display_img.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow("PPE Detection System", display_img)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if arduino_available:
    try:
        board.exit()
    except:
        pass

print("\nSystem stopped.")
print(f"Log saved to: {log_file}")