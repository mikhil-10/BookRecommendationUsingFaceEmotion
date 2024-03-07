# # import streamlit as st
# # import pandas as pd
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import linear_kernel
# # import random
# #
# # # Load the Kaggle datasets
# # kaggle_books_path = 'C:/Users/MIKHIL/OneDrive/Desktop/Projects/Bookmusic recommd/Bookmusic recommd/booksmain.csv'
# # kaggle_music_path = 'C:/Users/MIKHIL/OneDrive/Desktop/Projects/Bookmusic recommd/Bookmusic recommd/music.csv'
# #
# # try:
# #     kaggle_books_df = pd.read_csv(kaggle_books_path)
# #     kaggle_music_df = pd.read_csv(kaggle_music_path)
# # except pd.errors.ParserError as e:
# #     st.error(f"Error parsing CSV file: {e}")
# #     st.stop()
# #
# #
# # # Function to get book recommendations based on emotion
# # def get_book_recommendations(emotion):
# #     print("Filtering books...")
# #     emotion_df = kaggle_books_df[kaggle_books_df['Emotion'] == emotion]
# #     print("Filtered books count:", len(emotion_df))
# #     return emotion_df['title'].tolist()
# #
# # # Function to get movie recommendations based on emotion
# # def get_movie_recommendations(emotion):
# #     print("Filtering movies...")
# #     emotion_df = kaggle_music_df[kaggle_music_df['Emotion'] == emotion]
# #     print("Filtered movies count:", len(emotion_df))
# #     return emotion_df['title'].tolist()
# #
# # # Function to get content-based recommendations based on description similarity
# # def get_content_based_recommendations(selected_book):
# #     tfidf_vectorizer = TfidfVectorizer(stop_words="english")
# #     tfidf_matrix = tfidf_vectorizer.fit_transform(kaggle_books_df['description'].fillna(''))
# #
# #     book_index = kaggle_books_df[kaggle_books_df['title'] == selected_book].index[0]
# #     cosine_similarities = linear_kernel(tfidf_matrix[book_index], tfidf_matrix).flatten()
# #     similar_books_indices = cosine_similarities.argsort()[:-4:-1]
# #
# #     recommended_books = [kaggle_books_df.iloc[i]['title'] for i in similar_books_indices[1:]]  # Exclude the selected book
# #     return recommended_books
# #
# # # Streamlit UI
# # st.title("Rhythmic Reads Hub Chatbot")
# #
# # # Prompt the user to select Fictional or Non-Fictional
# # book_type = st.radio("Select Book Type:", ("Fictional", "Non-Fictional"))
# #
# # print("Selected Book Type:", book_type)
# #
# # # Filter emotions based on the selected book type
# # if book_type == "Fictional":
# #     filtered_emotions = ['sad', 'happy', 'thriller', 'horror']
# # elif book_type == "Non-Fictional":
# #     filtered_emotions = ['Business', 'Psychology', 'Biography']
# # else:
# #     filtered_emotions = []
# #
# # print("Filtered Emotions:", filtered_emotions)
# #
# # # Let's use st.selectbox to select an emotion
# # selected_emotion = st.selectbox("Select Emotion:", filtered_emotions)
# #
# # print("Selected Emotion:", selected_emotion)
# #
# # # Get recommended books based on the selected emotion
# # recommended_books = get_book_recommendations(selected_emotion)
# #
# # print("Recommended Books:", recommended_books)
# #
# # # Provide recommended books based on the selected emotion
# # st.header("Recommended Books:")
# #
# # # Check if there are enough books to sample
# # if len(recommended_books) >= 5:
# #     # Sample 5 random books
# #     sampled_books = random.sample(recommended_books, 5)
# #
# #     # Display sampled books
# #     for book in sampled_books:
# #         st.write(book)
# #
# #     # Get movie recommendation based on the selected emotion
# #     movie_recommendation = get_movie_recommendations(selected_emotion)
# #
# #     print("Movie Recommendation:", movie_recommendation)
# #
# #     # Display movie recommendation
# #     st.header("Movie Recommendation:")
# #     sampled_movies = random.sample(movie_recommendation, 5)
# #     for movie in sampled_movies:
# #         st.write(movie)
# #
# # else:
# #     st.warning("Select which type of book you want to read.")
#
#
#
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel
# import random
#
# # Load the Kaggle datasets
# kaggle_books_path = 'C:/Users/MIKHIL/OneDrive/Desktop/Projects/Bookmusic recommd/Bookmusic recommd/booksmain.csv'
# kaggle_music_path = 'C:/Users/MIKHIL/OneDrive/Desktop/Projects/Bookmusic recommd/Bookmusic recommd/musicmain.csv'
#
# try:
#     kaggle_books_df = pd.read_csv(kaggle_books_path)
#     kaggle_music_df = pd.read_csv(kaggle_music_path)
# except pd.errors.ParserError as e:
#     st.error(f"Error parsing CSV file: {e}")
#     st.stop()
#
#
# # Function to get book recommendations based on emotion
# def get_book_recommendations(emotion):
#     print("Filtering books...")
#     emotion_df = kaggle_books_df[kaggle_books_df['Emotion'] == emotion]
#     print("Filtered books count:", len(emotion_df))
#     return emotion_df['title'].tolist()
#
# # Function to get movie recommendations based on emotion
# def get_music_recommendations(emotion):
#     print("Filtering movies...")
#     emotion_df = kaggle_music_df[kaggle_music_df['Emotion'] == emotion]
#     print("Filtered movies count:", len(emotion_df))
#     return emotion_df['track_name'].tolist()
#
# # Function to get content-based recommendations based on description similarity
# def get_content_based_recommendations(selected_book):
#     tfidf_vectorizer = TfidfVectorizer(stop_words="english")
#     tfidf_matrix = tfidf_vectorizer.fit_transform(kaggle_books_df['description'].fillna(''))
#
#     book_index = kaggle_books_df[kaggle_books_df['title'] == selected_book].index[0]
#     cosine_similarities = linear_kernel(tfidf_matrix[book_index], tfidf_matrix).flatten()
#     similar_books_indices = cosine_similarities.argsort()[:-4:-1]
#
#     recommended_books = [kaggle_books_df.iloc[i]['title'] for i in similar_books_indices[1:]]  # Exclude the selected book
#     return recommended_books
#
# # Streamlit UI
# st.title("Rhythmic Reads Hub Chatbot")
#
# # Prompt the user to select Fictional or Non-Fictional
# book_type = st.radio("Select Book Type:", ("Fictional", "Non-Fictional"))
#
# print("Selected Book Type:", book_type)
#
# # Filter emotions based on the selected book type
# if book_type == "Fictional":
#     filtered_emotions = ['sad', 'happy', 'thriller', 'horror']
# elif book_type == "Non-Fictional":
#     filtered_emotions = ['Business', 'Psychology', 'Biography']
# else:
#     filtered_emotions = []
#
# print("Filtered Emotions:", filtered_emotions)
#
# # Let's use st.selectbox to select an emotion
# selected_emotion = st.selectbox("Select Emotion:", filtered_emotions)
#
# print("Selected Emotion:", selected_emotion)
#
# # Get recommended books based on the selected emotion
# recommended_books = get_book_recommendations(selected_emotion)
#
# print("Recommended Books:", recommended_books)
#
# # Provide recommended books based on the selected emotion
# st.header("Recommended Books:")
#
# # Check if there are enough books to sample
# if len(recommended_books) >= 5:
#     # Sample 5 random books
#     sampled_books = random.sample(recommended_books, 5)
#
#     # Display sampled books
#     for book in sampled_books:
#         st.write(book)
#
#     # Get movie recommendation based on the selected emotion
#     movie_recommendation = get_music_recommendations(selected_emotion)
#
#     print("Movie Recommendation:", movie_recommendation)
#
#     # Display movie recommendation
#     st.header("Music Recommendation:")
#     sampled_movies = random.sample(movie_recommendation, 5)
#     for movie in sampled_movies:
#         st.write(movie)
#
# else:
#     st.warning("Select which type of book you want to read.")


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random

# Load the Kaggle datasets
kaggle_books_path = 'C:/Users/MIKHIL/OneDrive/Desktop/Projects/Bookmusic recommd/Bookmusic recommd/booksmain.csv'
kaggle_music_path = 'C:/Users/MIKHIL/OneDrive/Desktop/Projects/Bookmusic recommd/Bookmusic recommd/musicmain.csv'

try:
    kaggle_books_df = pd.read_csv(kaggle_books_path)
    kaggle_music_df = pd.read_csv(kaggle_music_path)
except pd.errors.ParserError as e:
    st.error(f"Error parsing CSV file: {e}")
    st.stop()


# Function to get book recommendations based on emotion
def get_book_recommendations(emotion):
    print("Filtering books...")
    emotion_df = kaggle_books_df[kaggle_books_df['Emotion'] == emotion]
    print("Filtered books count:", len(emotion_df))
    return emotion_df['title'].tolist()

# Function to get movie recommendations based on emotion
def get_music_recommendations(emotion):
    print("Filtering movies...")
    emotion_df = kaggle_music_df[kaggle_music_df['Emotion'] == emotion]
    print("Filtered movies count:", len(emotion_df))
    return emotion_df['track_name'].tolist()

# Function to get content-based recommendations based on description similarity
def get_content_based_recommendations(selected_book):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(kaggle_books_df['description'].fillna(''))

    book_index = kaggle_books_df[kaggle_books_df['title'] == selected_book].index[0]
    cosine_similarities = linear_kernel(tfidf_matrix[book_index], tfidf_matrix).flatten()
    similar_books_indices = cosine_similarities.argsort()[:-4:-1]

    recommended_books = [kaggle_books_df.iloc[i]['title'] for i in similar_books_indices[1:]]  # Exclude the selected book
    return recommended_books

# Streamlit UI
st.title("Rhythmic Reads Hub Chatbot")

# Prompt the user to select Fictional or Non-Fictional
book_type = st.radio("Select Book Type:", (None, "Fictional", "Non-Fictional"))

print("Selected Book Type:", book_type)

# Filter emotions based on the selected book type
if book_type == "Fictional":
    filtered_emotions = ['sad', 'happy', 'thriller', 'horror']
elif book_type == "Non-Fictional":
    filtered_emotions = ['Business', 'Psychology', 'Biography']
else:
    filtered_emotions = []

print("Filtered Emotions:", filtered_emotions)

# Let's use st.selectbox to select an emotion
selected_emotion = st.selectbox("Select Emotion:", [None] + filtered_emotions)

print("Selected Emotion:", selected_emotion)

if selected_emotion:
    # Get recommended books based on the selected emotion
    recommended_books = get_book_recommendations(selected_emotion)

    print("Recommended Books:", recommended_books,"▪️")
    # Provide recommended books based on the selected emotion

    st.header("Recommended Books:")

    # Check if there are enough books to sample
    if len(recommended_books) >= 5:
        # Sample 5 random books
        sampled_books = random.sample(recommended_books, 5)

        # Display sampled books
        for book in sampled_books:
            st.write(book)

        # Get movie recommendation based on the selected emotion
        movie_recommendation = get_music_recommendations(selected_emotion)

        print("Movie Recommendation:", movie_recommendation)

        # Display movie recommendation
        st.header("Music Recommendation:")
        sampled_movies = random.sample(movie_recommendation, 5)
        for movie in sampled_movies:
            st.write(movie)

    else:
        st.warning("Select which type of book you want to read.")
