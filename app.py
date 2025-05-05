import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# Sample Movie Data
movies = pd.DataFrame({
    'title': ['Inception', 'Titanic', 'The Matrix', 'The Godfather', 'Avengers'],
    'genres': [['Action', 'Sci-Fi'], ['Romance', 'Drama'], ['Sci-Fi', 'Action'], ['Crime', 'Drama'], ['Action', 'Fantasy']],
    'rating': [8.8, 7.8, 8.7, 9.2, 8.4]
})

# Prepare genre vectors
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genres'])

# Genre list for selection
genre_choices = sorted(set(g for sublist in movies['genres'] for g in sublist))

# App title
st.title("🎬 Movie Recommender System")

# User inputs
selected_genres = st.multiselect("Choose your favorite genres", genre_choices)
top_n = st.slider("Number of Recommendations", min_value=1, max_value=5, value=3)

# Recommend function
def recommend_movie(user_genres, top_n=3):
    if not user_genres:
        return None
    
    user_vector = mlb.transform([user_genres])
    scores = cosine_similarity(user_vector, genre_matrix)[0]
    
    top_indices = np.argsort(scores)[::-1][:top_n]
    recommended = movies.iloc[top_indices][['title', 'genres', 'rating']]
    recommended['genres'] = recommended['genres'].apply(lambda g: ', '.join(g))
    return recommended

# Display recommendations
if selected_genres:
    recommendations = recommend_movie(selected_genres, top_n)
    st.subheader("Recommended Movies:")
    st.table(recommendations)
else:
    st.info("👈 Please select at least one genre to get recommendations.")
