# Deepfake Detection Meso4 Django

![Deepfake Detection](https://chiefit.me/wp-content/uploads/2024/05/Deepfake-Detection-Techniques.jpg)  <!-- Replace with an actual image link if available -->

## Overview

This project implements a deepfake detection system using the Meso4 deep learning architecture, built with Django. The application allows users to upload images and videos to detect potential deepfake content by analyzing individual frames through a convolutional neural network (CNN).

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Deep Learning Architecture](#deep-learning-architecture)
- [Program Architecture](#program-architecture)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Image and Video Upload**: Users can upload images and videos for deepfake detection.
- **Frame Analysis**: The system analyzes each frame in uploaded videos to detect deepfakes.
- **User Feedback**: Detected frames are returned to the user for review, along with messages indicating the detection status.

## Installation

To set up the project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/li812/Deepfake-Detection-Meso4-Django.git
   cd Deepfake-Detection-Meso4-Django
   ```

2. **Set up Anaconda environment:**
   ```bash
   conda create -n deepfake-detection python=3.8
   conda activate deepfake-detection
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

5. **Start the development server:**
   ```bash
   python manage.py runserver
   ```

6. **Access the application:**  
   Open your browser and go to `http://127.0.0.1:8000/`.

## Usage

1. Navigate to the application in your browser.
2. Upload an image or video using the provided upload button.
3. Wait for the model to process the file.
4. Review the detected fake frames or messages provided by the application.

## Deep Learning Architecture

### Meso4 Model

The Meso4 model is a convolutional neural network (CNN) specifically designed for deepfake detection. It utilizes a series of convolutional, batch normalization, and pooling layers to extract features from input images. The architecture is as follows:

- **Input Layer**: Accepts images of shape (256, 256, 3).
- **Convolutional Layers**: 
  - Four convolutional layers with increasing filter sizes, followed by batch normalization and max pooling layers to reduce dimensionality and extract relevant features.
  - Convolutional layer 1: 8 filters of size (3, 3)
  - Convolutional layer 2: 8 filters of size (5, 5)
  - Convolutional layer 3: 16 filters of size (5, 5)
  - Convolutional layer 4: 16 filters of size (5, 5)
- **Dropout Layers**: Applied to mitigate overfitting during training.
- **Fully Connected Layer**: A dense layer that outputs the final classification score.
- **Output Layer**: A sigmoid activation function outputs a probability score between 0 and 1, indicating the likelihood of a deepfake.

The architecture allows the model to learn complex patterns and features, making it effective for distinguishing between genuine and manipulated images or videos.

### Meso4 Class Implementation

```python
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
```

## Program Architecture

The application is structured using Django's Model-View-Template (MVT) architecture, with the following key components:

- **Models**: Define the data structure and handle the storage of uploaded files and detected frames.
- **Views**: Handle the logic for processing user requests, loading the Meso4 model, predicting fake frames, and rendering the results back to the user.
- **Templates**: HTML files that define the structure and layout of the user interface, providing a seamless experience for file uploads and results display.
- **Static Files**: CSS and JavaScript files that enhance the styling and interactivity of the application.

The application also utilizes TensorFlow for deep learning tasks, with a well-defined workflow for loading models, preprocessing images, and making predictions.

## Screenshots

![Deepfake Detection Screenshot 1](Screenshots/Deepfake%20Detection1.png)
![Deepfake Detection Screenshot 2](Screenshots/Deepfake%20Detection2.png)

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
