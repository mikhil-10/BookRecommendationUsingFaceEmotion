# import cv2
# import numpy as np
# import streamlit as st
# import pandas as pd
# from keras.models import model_from_json
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# import random
#
# # Load the Kaggle datasets
# kaggle_books_path = 'C:/Users/MIKHIL/OneDrive/Desktop/Projects/BookRecommendationUsingFaceEmotion/Bookmusic recommd/booksmain.csv'
# kaggle_music_path = 'C:/Users/MIKHIL/OneDrive/Desktop/Projects/BookRecommendationUsingFaceEmotion/Bookmusic recommd/musicmain.csv'
#
# try:
#     kaggle_books_df = pd.read_csv(kaggle_books_path)
#     kaggle_music_df = pd.read_csv(kaggle_music_path)
# except pd.errors.ParserError as e:
#     st.error(f"Error parsing CSV file: {e}")
#     st.stop()
#
# # Load the emotion detection model
# json_file = open("facialemotionmodel.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights("facialemotionmodel.h5")
# haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(haar_file)
#
# # Function to extract facial features and detect emotion
# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1, 48, 48, 1)
#     return feature / 255.0
#
# # Function to detect emotion from the captured image
# def detect_emotion(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(image, 1.3, 5)
#     emotions = []
#     try:
#         for (p, q, r, s) in faces:
#             face_image = gray[q:q + s, p:p + r]
#             face_image = cv2.resize(face_image, (48, 48))
#             img = extract_features(face_image)
#             pred = model.predict(img)
#             emotion_label = np.argmax(pred)  # Assuming the model returns emotion labels as indices
#             emotions.append(emotion_label)
#         return emotions
#     except cv2.error:
#         return None
#
# # Function to get book recommendations based on emotion
# def get_book_recommendations(emotion):
#     emotion_df = kaggle_books_df[kaggle_books_df['Emotion'] == emotion]
#     return emotion_df['title'].tolist()
#
# # Function to get music recommendations based on emotion
# def get_music_recommendations(emotion):
#     emotion_df = kaggle_music_df[kaggle_music_df['Emotion'] == emotion]
#     return emotion_df['track_name'].tolist()
#
# # Convert emotion labels to the corresponding emotions
# EMOTIONS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
#
# # Streamlit UI
# st.title("Rhythmic Reads Hub Chatbot")
#
# # Add a button to capture image
# if st.button("Capture Image"):
#     # Capture image using webcam
#     webcam = cv2.VideoCapture(0)
#
#     # Capture and display the image
#     ret, frame = webcam.read()
#     cv2.imshow("Captured Image", frame)
#     webcam.release()
#     cv2.destroyAllWindows()
#
#     # Detect emotion from the captured image
#     captured_emotions = detect_emotion(frame)
#
#     if captured_emotions:
#         # Convert emotion labels to emotions
#         detected_emotions = [EMOTIONS[label] for label in captured_emotions]
#
#         # Display detected emotions
#         st.header("Detected Emotion:")
#         st.write(detected_emotions[0])
#
#         # Get book recommendation based on the first detected emotion
#         book_recommendation = []
#         for emotion in detected_emotions:
#             book_recommendation += get_book_recommendations(emotion)
#
#         # Display 5 random book recommendations
#         st.header("Recommended Books:")
#         if book_recommendation:
#             random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
#             for book in random_books:
#                 st.write(book)
#         else:
#             st.write("No books found for the detected emotion.")
#
#         # Get music recommendation based on the first detected emotion
#         music_recommendation = []
#         for emotion in detected_emotions:
#             music_recommendation += get_music_recommendations(emotion)
#
#         # Randomly select 5 music tracks
#         st.header("Music Recommendation:")
#         if music_recommendation:
#             random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
#             for music in random_music:
#                 st.write(music)
#         else:
#             st.write("No music found for the detected emotion.")
#     else:
#         st.error("No faces detected in the captured image.")


import cv2
import numpy as np
import streamlit as st
import pandas as pd
from keras.models import model_from_json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random
import time

# Load the Kaggle datasets
kaggle_books_path = 'C:/Users/MIKHIL/OneDrive/Desktop/Projects/BookRecommendationUsingFaceEmotion/Bookmusic recommd/booksmain.csv'
kaggle_music_path = 'C:/Users/MIKHIL/OneDrive/Desktop/Projects/BookRecommendationUsingFaceEmotion/Bookmusic recommd/musicmain.csv'

try:
    kaggle_books_df = pd.read_csv(kaggle_books_path)
    kaggle_music_df = pd.read_csv(kaggle_music_path)
except pd.errors.ParserError as e:
    st.error(f"Error parsing CSV file: {e}")
    st.stop()

# Load the emotion detection model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract facial features and detect emotion
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to detect emotion from the captured image
def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    emotions = []
    try:
        for (p, q, r, s) in faces:
            face_image = gray[q:q + s, p:p + r]
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)
            pred = model.predict(img)
            emotion_label = np.argmax(pred)  # Assuming the model returns emotion labels as indices
            emotions.append(emotion_label)
        return emotions
    except cv2.error:
        return None

# Function to get book recommendations based on emotion
def get_book_recommendations(emotion):
    emotion_df = kaggle_books_df[kaggle_books_df['Emotion'] == emotion]
    return emotion_df['title'].tolist()

# Function to get music recommendations based on emotion
def get_music_recommendations(emotion):
    emotion_df = kaggle_music_df[kaggle_music_df['Emotion'] == emotion]
    return emotion_df['track_name'].tolist()

# Convert emotion labels to the corresponding emotions
EMOTIONS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Streamlit UI
st.title("Rhythmic Reads Hub Chatbot")

# Placeholder for status message
status_placeholder = st.empty()

# Add a button to capture image
if st.button("Capture Image"):
    # Display message: Getting ready to capture image...
    status_placeholder.text("Getting ready to capture image...")

    # Give a delay of 2 seconds
    time.sleep(2)

    # Capture image using webcam
    webcam = cv2.VideoCapture(0)

    # Capture and display the image
    ret, frame = webcam.read()
    cv2.imshow("Captured Image", frame)
    webcam.release()
    cv2.destroyAllWindows()

    # Update status message: Image captured successfully
    status_placeholder.text("Image captured successfully.\nTo Re-capture image click on Capture Image.")

    # Detect emotion from the captured image
    captured_emotions = detect_emotion(frame)

    if captured_emotions:
        # Convert emotion labels to emotions
        detected_emotions = [EMOTIONS[label] for label in captured_emotions]

        # Display detected emotions
        st.header("Detected Emotion:")
        st.write(detected_emotions[0])


        # Get book recommendation based on the first detected emotion
        book_recommendation = []
        for emotion in detected_emotions:
            book_recommendation += get_book_recommendations(emotion)

        # Display 5 random book recommendations
        st.header("Recommended Books based on Emotion:")
        if book_recommendation:
            random_books = random.sample(list(set(book_recommendation)), min(5, len(book_recommendation)))
            for book in random_books:
                st.write(book)
        else:
            st.write("No books found for the detected emotions.")

        # Get music recommendation based on the first detected emotion
        music_recommendation = []
        for emotion in detected_emotions:
            music_recommendation += get_music_recommendations(emotion)

        # Display music recommendation
        st.header("Music Recommendation:")
        if music_recommendation:
            random_music = random.sample(list(set(music_recommendation)), min(5, len(music_recommendation)))
            for music in random_music:
                st.write(music)
        else:
            st.write("No music found for the detected emotions.")
    else:
        st.error("No faces detected in the captured image.")

