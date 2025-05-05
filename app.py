# Content-Based Movie Recommender using Streamlit

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load movie data
@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')  # Must include: movieId, title, genres
    movies['genres'] = movies['genres'].fillna('')
    return movies

movies = load_data()

# TF-IDF vectorization on genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Mapping titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def get_content_recommendations(title, n=5):
    if title not in indices:
        return pd.DataFrame(columns=['title', 'genres'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]

# Streamlit UI
st.title("🎬 AI-Powered Movie Recommender")
st.write("Get movie recommendations based on genre similarity.")

selected_movie = st.selectbox("Select a movie you like:", sorted(movies['title'].unique()))
top_n = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)

if selected_movie:
    results = get_content_recommendations(selected_movie, top_n)
    st.subheader(f"🎥 Movies similar to *{selected_movie}*:")
    st.table(results)
