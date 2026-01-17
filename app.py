import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from PIL import Image

# Emotion labels
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

# Build model (same architecture as training)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load trained weights
model.load_weights("model.h5")

def predict_emotion(image):
    if image is None:
        return "Please upload an image"

    image = image.convert("L")  # grayscale
    img = np.array(image)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = img.reshape(1, 48, 48, 1)

    prediction = model.predict(img)
    emotion = emotion_dict[np.argmax(prediction)]
    return emotion

interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=gr.Label(label="Predicted Emotion"),
    title="Emotion Detection",
    description="Upload a face image and detect emotion using CNN"
)

interface.launch()
