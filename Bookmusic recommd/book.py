import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Define the book and music recommendation data
fictional_books = {
    "Happy": ["Book1", "Book2", "Book3"],
    "Sad": ["Book4", "Book5", "Book6"],
    "Thriller": ["Book7", "Book8", "Book9"],
    "Horror": ["Book10", "Book11", "Book12"],
}

non_fictional_books = {
    "Psychology": ["Book13", "Book14", "Book15"],
    "Biography": ["Book16", "Book17", "Book18"],
    "Business": ["Book19", "Book20", "Book21"],
    "Education": ["Book22", "Book23", "Book24"],
}

music_recommendations = {
    "Happy": "HappySong1",
    "Sad": "SadSong1",
    "Thriller": "ThrillerSong1",
    "Horror": "HorrorSong1",
    "Psychology": "PsychologySong1",
    "Biography": "BiographySong1",
    "Business": "BusinessSong1",
    "Education": "EducationSong1",
}


# Function to get book recommendations based on emotion
def get_book_recommendations(genre, emotion):
    books = (
        fictional_books.get(emotion, [])
        if genre == "Fictional"
        else non_fictional_books.get(emotion, [])
    )
    return books


# Function to get music recommendation based on emotion
def get_music_recommendation(emotion):
    return music_recommendations.get(emotion, "")


# Function to get book recommendations based on description similarity
def get_content_based_recommendations(selected_book):
    all_books = []
    for genre, books in fictional_books.items():
        all_books.extend(books)
    for genre, books in non_fictional_books.items():
        all_books.extend(books)

    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_books)

    book_index = all_books.index(selected_book)
    cosine_similarities = linear_kernel(
        tfidf_matrix[book_index], tfidf_matrix
    ).flatten()
    similar_books_indices = cosine_similarities.argsort()[:-4:-1]

    recommended_books = [
        all_books[i] for i in similar_books_indices[1:]
    ]  # Exclude the selected book
    return recommended_books


# Streamlit UI
st.title("Rhythmic Reads Hub Chatbot")

# Let's use st.text_input to simulate a chat input
user_input = st.text_input("You: Type 'Books'").lower()

# Process the user input
if "book" in user_input:
    st.write("Fictional or Non-Fictional?")
    book_genre = st.radio("Select Book Genre:", ("Fictional", "Non-Fictional"))

    if book_genre == "Fictional":
        st.write("Select Fictional Book Emotion:")
        fiction_emotion = st.radio("", ("Happy", "Sad", "Thriller", "Horror"))
        recommended_books = get_book_recommendations("Fictional", fiction_emotion)
    elif book_genre == "Non-Fictional":
        st.write("Select Non-Fictional Book Emotion:")
        non_fiction_emotion = st.radio(
            "",
            ("Psychology", "Biography", "Business", "Education"),
        )
        recommended_books = get_book_recommendations(
            "Non-Fictional", non_fiction_emotion
        )

    st.write(f"Recommended Books: {recommended_books}")

    # Display music recommendation based on selected book emotion
    selected_emotion = (
        fiction_emotion if book_genre == "Fictional" else non_fiction_emotion
    )
    music_recommendation = get_music_recommendation(selected_emotion)
    st.write(f"Music Recommendation: {music_recommendation}")

    # Content-based recommendations
    st.write("Select a book for more recommendations:")
    selected_book = st.selectbox("", recommended_books)
    content_based_recommendations = get_content_based_recommendations(selected_book)
    st.write(f"Content-Based Recommendations: {content_based_recommendations}")
