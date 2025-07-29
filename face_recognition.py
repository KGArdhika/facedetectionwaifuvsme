# face_recognition.py
import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import urllib.request
import zipfile


class FaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.label_encoder = LabelEncoder()
        self.face_size = (128, 128)
        
    def collect_dataset(self, person_name, num_images=100):
        """Mengumpulkan dataset wajah dari webcam"""
        cap = cv2.VideoCapture(0)
        dataset_path = f'dataset/{person_name}'
        
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            
        count = 0
        print(f"Collecting {num_images} images for {person_name}...")
        
        while count < num_images:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, self.face_size)
                
                # Save face image
                cv2.imwrite(f'{dataset_path}/{person_name}_{count}.jpg', face_resized)
                count += 1
                
            cv2.putText(frame, f'Images collected: {count}/{num_images}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Collecting Face Data', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def load_mit_dataset(self):
        """Download dan load MIT face dataset"""
        if not os.path.exists('dataset/MIT_faces'):
            print("Downloading MIT face dataset...")
            # Simulasi download - ganti dengan URL yang sebenarnya
            # urllib.request.urlretrieve('URL_DATASET', 'mit_faces.zip')
            # with zipfile.ZipFile('mit_faces.zip', 'r') as zip_ref:
            #     zip_ref.extractall('dataset/MIT_faces')
            
    def prepare_data(self):
        """Mempersiapkan data untuk training"""
        data = []
        labels = []
        
        for person in os.listdir('dataset'):
            person_path = os.path.join('dataset', person)
            if os.path.isdir(person_path):
                for image_name in os.listdir(person_path):
                    image_path = os.path.join(person_path, image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        image = cv2.resize(image, self.face_size)
                        data.append(image)
                        labels.append(person)
                        
        data = np.array(data).reshape(-1, self.face_size[0], self.face_size[1], 1)
        data = data.astype('float32') / 255.0
        
        labels = self.label_encoder.fit_transform(labels)
        labels = to_categorical(labels)
        
        return train_test_split(data, labels, test_size=0.2, random_state=42)
    
    def build_model(self, num_classes):
        """Membangun model CNN untuk face recognition"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.face_size[0], self.face_size[1], 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train_model(self):
        """Train the face recognition model"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        num_classes = y_train.shape[1]
        
        self.model = self.build_model(num_classes)
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        
        datagen.fit(X_train)
        
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=50,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        self.model.save('face_recognition_model.h5')
        np.save('label_encoder.npy', self.label_encoder.classes_)
        
        return history
    
    def load_trained_model(self):
        """Load model yang sudah ditraining"""
        self.model = load_model('face_recognition_model.h5')
        self.label_encoder.classes_ = np.load('label_encoder.npy')
        
    def get_face_embedding(self, face_image):
        """Extract face embedding dari model"""
        face_resized = cv2.resize(face_image, self.face_size)
        face_normalized = face_resized.reshape(1, self.face_size[0], self.face_size[1], 1).astype('float32') / 255.0
        
        # Get embedding dari layer sebelum output
        intermediate_model = Sequential(self.model.layers[:-1])
        embedding = intermediate_model.predict(face_normalized)
        
        return embedding.flatten()
    
    def recognize_face(self, face_image):
        """Recognize face dan return nama + confidence"""
        face_resized = cv2.resize(face_image, self.face_size)
        face_normalized = face_resized.reshape(1, self.face_size[0], self.face_size[1], 1).astype('float32') / 255.0
        
        predictions = self.model.predict(face_normalized)
        confidence = np.max(predictions)
        
        if confidence > 0.7:  # Threshold confidence
            label_idx = np.argmax(predictions)
            name = self.label_encoder.inverse_transform([label_idx])[0]
            return name, confidence
        else:
            return "Unknown", confidence
    
    def real_time_recognition(self):
        """Real-time face recognition dari webcam"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                name, confidence = self.recognize_face(face)
                
                # Draw rectangle dan label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{name} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, color, 2)
                
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
    face_system = FaceRecognitionSystem()
    
    while True:
        print("\n=== Face Recognition System ===")
        print("1. Collect face data")
        print("2. Train model")
        print("3. Real-time recognition")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            name = input("Enter person name: ")
            face_system.collect_dataset(name)
        elif choice == '2':
            print("Training model...")
            face_system.train_model()
            print("Training completed!")
        elif choice == '3':
            print("Loading model...")
            face_system.load_trained_model()
            print("Starting real-time recognition... Press 'q' to quit")
            face_system.real_time_recognition()
        elif choice == '4':
            break