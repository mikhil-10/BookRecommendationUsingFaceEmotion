import cv2
from keras.models import model_from_json
import numpy as np

# **Replace these placeholders with your actual implementation:**
def load_model_and_classifier():
    """
    Loads the pre-trained facial emotion recognition model and classifier.

    Raises:
        FileNotFoundError: If the model files are not found.

    Returns:
        tuple: A tuple containing the loaded model and classifier.
    """

    model_json_path = "facialemotionmodel.json"  # Replace with your model file path
    model_weights_path = "facialemotionmodel.h5"  # Replace with your model weights file path
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # Replace with your Haar cascade file path

    try:
        # Load the model architecture
        with open(model_json_path, "r") as f:
            model_json = f.read()
        model = model_from_json(model_json)

        # Load the model weights
        model.load_weights(model_weights_path)

        # Load the Haar cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)

        return model, face_cascade
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Model files not found. {e}")

def extract_features(image):
    """
    Preprocesses an image for use with the facial emotion recognition model.

    Args:
        image (np.ndarray): The image to be preprocessed.

    Returns:
        np.ndarray: The preprocessed image.
    """

    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape to match the model's input format
    return feature / 255.0  # Normalize pixel values

def detect_emotion(webcam):
    """
    Detects the emotion in a frame captured from the webcam.

    Args:
        webcam (cv2.VideoCapture): The webcam object.

    Returns:
        str: The detected emotion, or None if no face is detected.
    """

    try:
        model, face_cascade = load_model_and_classifier()
        labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
                 4: "neutral", 5: "sad", 6: "surprise"}

        while True:
            ret, frame = webcam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                image = gray[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                predicted_emotion = labels[pred.argmax()]
                return predicted_emotion  # Return the detected emotion

            # If no face is detected
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None  # Indicate error or no face detected

if __name__ == "__main__":
    webcam = cv2.VideoCapture(0)
    detected_emotion = detect_emotion(webcam)
    webcam.release()
    cv2.destroyAllWindows()

    if detected_emotion:
        print(f"Detected emotion: {detected_emotion}")
    else:
        print("No face detected.")
