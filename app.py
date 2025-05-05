import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# Set up Streamlit config
st.set_page_config(page_title="AI Movie Recommender", layout="centered")

# Load data from a working GitHub link
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/gopala-kr/Movielens-Dataset-Exploration/master/movies.csv"
    df = pd.read_csv(url)
    df = df.dropna(subset=['genres'])  # Drop rows with no genre
    df['genres'] = df['genres'].apply(lambda x: x.split('|'))  # Split genres into lists
    df = df[df['genres'].map(len) > 0]  # Remove rows with empty genre lists
    df['rating'] = np.random.uniform(6.0, 9.5, len(df))  # Fake ratings for now
    return df

movies = load_data()

# Extract all genres
all_genres = sorted(set(genre for sublist in movies['genres'] for genre in sublist))

# Recommendation engine
def recommend_movie(user_genres, top_n):
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies['genres'])
    user_vector = mlb.transform([user_genres])
    scores = cosine_similarity(user_vector, genre_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    recommended = movies.iloc[top_indices][['title', 'genres', 'rating']].copy()
    recommended['genres'] = recommended['genres'].apply(lambda g: ', '.join(g))
    return recommended

# UI
st.title("🎬 AI-Powered Movie Recommender")
st.markdown("Select your favorite genres to get personalized movie recommendations from the MovieLens dataset.")

selected_genres = st.multiselect("Choose your favorite genres:", all_genres)
top_n = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

if selected_genres:
    result_df = recommend_movie(selected_genres, top_n)
    st.subheader("🎥 Recommended Movies:")
    st.table(result_df)
else:
    st.info("👈 Please select at least one genre.")
