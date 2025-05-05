import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import ast

# Set Streamlit config
st.set_page_config(page_title="AI Movie Recommender", layout="centered")

# Load and merge data
@st.cache_data
def load_data():
    # Load datasets from GitHub
    movies_url = "https://raw.githubusercontent.com/mohankumar624/NM-AI-Movie-recommendation/main/tmdb_5000_movies.csv"
    credits_url = "https://raw.githubusercontent.com/mohankumar624/NM-AI-Movie-recommendation/main/tmdb_5000_credits.csv"

    movies_df = pd.read_csv(movies_url)
    credits_df = pd.read_csv(credits_url)

    # Rename and merge
    credits_df.rename(columns={'movie_id': 'id'}, inplace=True)
    movies = movies_df.merge(credits_df, on='id')

    # Parse genres
    def parse_genres(genre_str):
        try:
            genre_list = ast.literal_eval(genre_str)
            return [g['name'] for g in genre_list]
        except:
            return []

    movies['genres'] = movies['genres'].apply(parse_genres)
    return movies

# Load dataset
movies = load_data()

# Prepare genre vectors using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genres'])

# Genre list for UI
genre_choices = sorted(set(genre for sublist in movies['genres'] for genre in sublist))

# Recommend function
def recommend_movie(user_genres, top_n=5):
    if not user_genres:
        return pd.DataFrame()

    # Convert user input to binary vector
    user_vector = mlb.transform([user_genres])

    # Compute cosine similarity
    scores = cosine_similarity(user_vector, genre_matrix)[0]

    # Get top N indices
    top_indices = np.argsort(scores)[::-1][:top_n]

    # Build result
    recommended = movies.iloc[top_indices][['title', 'genres', 'vote_average']].copy()
    recommended['genres'] = recommended['genres'].apply(lambda g: ', '.join(g))
    recommended.rename(columns={'vote_average': 'Rating'}, inplace=True)
    return recommended

# Streamlit UI
def main():
    st.title("🎬 AI-Powered Movie Recommender")
    st.markdown("Select your favorite genres and get personalized movie suggestions.")

    # User Inputs
    selected_genres = st.multiselect("Choose your favorite genres:", genre_choices)
    top_n = st.slider("Number of movie recommendations:", 1, 10, 5)

    # Recommend
    if selected_genres:
        result_df = recommend_movie(selected_genres, top_n)
        st.subheader("🎥 Recommended Movies:")
        st.table(result_df)
    else:
        st.info("👈 Please select at least one genre to get recommendations.")

if __name__ == "__main__":
    main()
