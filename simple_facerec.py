import cv2
import os
import glob
import numpy as np
import pickle

class SimpleFacerec:
    def __init__(self):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.labels = {}
        self.model_file = "face_recognizer_model.yml"
        self.labels_file = "face_labels.pkl"
        
        # Resize frame for faster processing
        self.frame_resizing = 0.5

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path and train the model
        :param images_path: Path to images directory
        :return: True if successful, False otherwise
        """
        # Create images path if it doesn't exist
        if not os.path.exists(images_path):
            print(f"Warning: Images folder '{images_path}' not found.")
            os.makedirs(images_path, exist_ok=True)
            print(f"Created empty directory: {images_path}")
            return False
        
        # Get all image files
        images_path = glob.glob(os.path.join(images_path, "*", "*.*"))
        
        print(f"Found {len(images_path)} image files.")
        
        if not images_path:
            print("No images found. Please add images in the format: Images/PersonName/photo.jpg")
            return False
        
        # Organize images by person
        person_images = {}
        for img_path in images_path:
            # Get person name from directory
            person_name = os.path.basename(os.path.dirname(img_path))
            if person_name not in person_images:
                person_images[person_name] = []
            person_images[person_name].append(img_path)
        
        faces = []
        labels = []
        label_id = 0
        
        # Process each person's images
        for person_name, img_paths in person_images.items():
            print(f"Processing {person_name}: {len(img_paths)} images")
            self.labels[label_id] = person_name
            
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  Could not read image: {img_path}")
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces_detected = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces_detected) == 0:
                    print(f"  No face detected in: {os.path.basename(img_path)}")
                    continue
                
                # Use the largest face
                (x, y, w, h) = max(faces_detected, key=lambda rect: rect[2] * rect[3])
                
                # Resize face to standard size
                face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                faces.append(face_roi)
                labels.append(label_id)
            
            label_id += 1
        
        if not faces:
            print("No valid faces found for training.")
            return False
        
        # Train the face recognizer
        print(f"Training with {len(faces)} face samples from {len(self.labels)} persons...")
        self.face_recognizer.train(faces, np.array(labels))
        
        # Save the trained model
        self.face_recognizer.save(self.model_file)
        
        # Save labels
        with open(self.labels_file, 'wb') as f:
            pickle.dump(self.labels, f)
        
        print(f"Encoding images loaded and model saved to {self.model_file}")
        return True
    
    def load_trained_model(self):
        """
        Load a previously trained model
        :return: True if successful, False otherwise
        """
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
        """
        Detect and recognize faces in the frame
        :param frame: Input frame
        :return: face_locations, face_names
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        
        # Convert to grayscale
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_locations = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        face_names = []
        scaled_face_locations = []
        
        for (x, y, w, h) in face_locations:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Predict using trained model
            label, confidence = self.face_recognizer.predict(face_roi)
            
            # Determine name based on confidence
            confidence_threshold = 70  # Lower is better for LBPH
            
            if confidence < confidence_threshold and label in self.labels:
                name = self.labels[label]
                confidence_text = f" ({100-confidence:.1f}%)"
            else:
                name = "Unknown"
                confidence_text = ""
            
            face_names.append(name)
            
            # Scale coordinates back to original frame size
            x_orig = int(x / self.frame_resizing)
            y_orig = int(y / self.frame_resizing)
            w_orig = int(w / self.frame_resizing)
            h_orig = int(h / self.frame_resizing)
            
            scaled_face_locations.append((y_orig, y_orig + h_orig, x_orig + w_orig, x_orig))
            
            # Draw on frame (optional, for debugging)
            cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), 
                         (255, 255, 0), 2)
            cv2.putText(frame, f"{name}{confidence_text}", 
                       (x_orig, y_orig - 10), cv2.FONT_HERSHEY_DUPLEX, 
                       0.7, (255, 255, 0), 2)
        
        return scaled_face_locations, face_names
    
    def add_new_face(self, frame, name):
        """
        Add a new face to the training data
        :param frame: Frame containing the face
        :param name: Name of the person
        :return: True if successful, False otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces_detected) == 0:
            print("No face detected in the frame.")
            return False
        
        # Use the largest face
        (x, y, w, h) = max(faces_detected, key=lambda rect: rect[2] * rect[3])
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        
        # Add to training data
        if not hasattr(self, 'training_faces'):
            self.training_faces = []
            self.training_labels = []
        
        # Assign new label ID if new person
        if name not in self.labels.values():
            new_label = max(self.labels.keys()) + 1 if self.labels else 0
            self.labels[new_label] = name
            label_id = new_label
        else:
            # Find existing label ID
            label_id = [k for k, v in self.labels.items() if v == name][0]
        
        self.training_faces.append(face_roi)
        self.training_labels.append(label_id)
        
        print(f"Added face for {name}")
        return True
    
    def retrain_model(self):
        """
        Retrain the model with updated faces
        :return: True if successful, False otherwise
        """
        if not hasattr(self, 'training_faces') or not self.training_faces:
            print("No new faces to train.")
            return False
        
        # Combine with existing data if model exists
        if os.path.exists(self.model_file):
            self.face_recognizer.read(self.model_file)
        
        # Train with all data
        all_faces = self.training_faces
        all_labels = np.array(self.training_labels)
        
        if hasattr(self, 'existing_faces'):
            all_faces.extend(self.existing_faces)
            all_labels = np.concatenate([all_labels, self.existing_labels])
        
        self.face_recognizer.train(all_faces, all_labels)
        self.face_recognizer.save(self.model_file)
        
        # Save labels
        with open(self.labels_file, 'wb') as f:
            pickle.dump(self.labels, f)
        
        print(f"Model retrained and saved with {len(self.labels)} persons")
        return True