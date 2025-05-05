import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# Must be at the top — avoids page config errors
st.set_page_config(page_title="AI Movie Recommender", layout="centered")

# Sample Movie Data
movies = pd.DataFrame({
    'title': ['Inception', 'Titanic', 'The Matrix', 'The Godfather', 'Avengers'],
    'genres': [['Action', 'Sci-Fi'], ['Romance', 'Drama'], ['Sci-Fi', 'Action'], ['Crime', 'Drama'], ['Action', 'Fantasy']],
    'rating': [8.8, 7.8, 8.7, 9.2, 8.4]
})

# Genre list for UI
genre_choices = sorted(set(genre for sublist in movies['genres'] for genre in sublist))

def recommend_movie(user_genres, top_n):
    # Transform genres using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies['genres'])

    # Convert user input to vector
    user_vector = mlb.transform([user_genres])

    # Compute cosine similarity
    scores = cosine_similarity(user_vector, genre_matrix)[0]

    # Get top N indices
    top_indices = np.argsort(scores)[::-1][:top_n]

    # Prepare output
    recommended = movies.iloc[top_indices][['title', 'genres', 'rating']].copy()
    recommended['genres'] = recommended['genres'].apply(lambda g: ', '.join(g))
    return recommended

# UI Layout
st.title("🎬 AI-Powered Movie Recommender")
st.markdown("Select your favorite genres and get personalized movie suggestions.")

selected_genres = st.multiselect("Choose your favorite genres:", genre_choices)
top_n = st.slider("Number of movie recommendations:", min_value=1, max_value=5, value=3)

# Trigger recommendation
if selected_genres:
    result_df = recommend_movie(selected_genres, top_n)
    st.subheader("🎥 Recommended Movies:")
    st.table(result_df)
else:
    st.info("👈 Please select at least one genre to see recommendations.")
