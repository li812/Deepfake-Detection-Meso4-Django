import os
import numpy as np
import cv2
import shutil 
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Flatten, MaxPooling2D, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

# Define the Classifier class
class Classifier:
    def __init__(self) -> None:
        self.model = None

    def predict(self, x):
        return self.model.predict(x)

    def load(self, path):
        self.model.load_weights(path)

# Define the Meso4 model class
class Meso4(Classifier):
    def __init__(self, lr=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def init_model(self):
        x = Input(shape=(256, 256, 3))

        # Model architecture
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        # Output layer
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

# Function to predict fake frames
def predict_fake_frames(input_path, output_folder, model):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    deepfake_detected = False

    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            frame = cv2.imread(input_path)
            if frame is not None:
                frame = cv2.resize(frame, (256, 256))
                normalized_frame = frame.astype('float32') / 255.0
                input_frame = np.expand_dims(normalized_frame, axis=0)
                prediction = model.predict(input_frame)
                if prediction <= 0.8:
                    cv2.imwrite(f'{output_folder}/frame_{os.path.basename(input_path)}.jpg', frame)
                    deepfake_detected = True
        elif input_path.lower().endswith(('.mp4', '.avi', '.mkv')):
            cap = cv2.VideoCapture(input_path)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                resized_frame = cv2.resize(frame, (256, 256))
                normalized_frame = resized_frame.astype('float32') / 255.0
                input_frame = np.expand_dims(normalized_frame, axis=0)
                prediction = model.predict(input_frame)
                if prediction <= 0.7:
                    cv2.imwrite(f'{output_folder}/frame_{frame_count}.jpg', frame)
                    deepfake_detected = True
            cap.release()

    return deepfake_detected

def index(request):
    # Before detection, clear the directory if it exists
    output_folder = os.path.join(settings.MEDIA_ROOT, 'Detected_Fake_Frames')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        meso = Meso4()
        meso.load(os.path.join(settings.BASE_DIR, 'Weights/Meso4_DF.h5'))

        if predict_fake_frames(full_file_path, output_folder, meso):
            # Retrieve the correct path for each detected frame
            detected_frames = [os.path.join(settings.MEDIA_URL, 'Detected_Fake_Frames', f) for f in os.listdir(output_folder)]
            # Delete the uploaded file after detection
            os.remove(full_file_path)  # Add this line to delete the uploaded file
            return render(request, 'detection/index.html', {'detected_frames': detected_frames})
        else:
            # Delete the uploaded file if no frames are detected
            os.remove(full_file_path)  # Add this line to delete the uploaded file
            return render(request, 'detection/index.html', {'message': 'No deepfaked frames detected.'})

    return render(request, 'detection/index.html')
