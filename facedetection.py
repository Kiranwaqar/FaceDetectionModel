import cv2
import gradio as gr
import numpy as np


def detect_faces(input_image):
    # Convert to OpenCV format (BGR)
    image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Load Haar cascade model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to RGB for display
    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return output_image


# Gradio UI
demo = gr.Interface(
    fn=detect_faces,
    inputs=gr.Image(type="numpy"),  # Use NumPy array for better processing
    outputs="image",
    title="Face Detection",
    description="Upload an image and the model will detect faces using OpenCV.",
    allow_flagging="never"  
)

demo.launch()
