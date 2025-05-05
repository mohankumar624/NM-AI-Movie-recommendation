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
 
 # Prepare genre vectors
 mlb = MultiLabelBinarizer()
 genre_matrix = mlb.fit_transform(movies['genres'])
 
 # Genre list
 genre_choices = sorted(set(g for sublist in movies['genres'] for g in sublist))
 
 # Streamlit UI
 st.title("🎬 AI-Powered Movie Recommender")
 st.write("Get movie recommendations based on genre similarity.")
 def main():
     st.set_page_config(page_title="AI Movie Recommender", layout="centered")
     st.title("🎬 AI-Powered Movie Recommender")
     st.markdown("Select your favorite genres and get personalized movie suggestions.")
 
 selected_movie = st.selectbox("Select a movie you like:", sorted(movies['title'].unique()))
 top_n = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)
     # User Inputs
     selected_genres = st.multiselect("Choose your favorite genres:", genre_choices)
     top_n = st.slider("How many recommendations would you like?", min_value=1, max_value=5, value=3)
 
     # Recommendation logic
     if selected_genres:
         recommendations = recommend_movie(selected_genres, top_n)
         st.subheader("🎥 Recommended Movies:")
         st.table(recommendations)
     else:
         st.info("👈 Please select at least one genre to get recommendations.")
 
 # Recommend function
 def recommend_movie(user_genres, top_n=3):
     if not user_genres:
         return pd.DataFrame()
     
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
     recommended = movies.iloc[top_indices][['title', 'genres', 'rating']]
  
     # Prepare output
     recommended = movies.iloc[top_indices][['title', 'genres', 'rating']].copy()
     recommended['genres'] = recommended['genres'].apply(lambda g: ', '.join(g))
     return recommended
 
 if __name__ == "__main__":
     main()
 # UI Layout
 st.title("🎬 AI-Powered Movie Recommender")
 st.markdown("Select your favorite genres and get personalized movie suggestions.")
  
 selected_genres = st.multiselect("Choose your favorite genres:", genre_choices)
 top_n = st.slider("Number of movie recommendations:", min_value=1, max_value=5, value=3)
 
 if selected_movie:
     results = get_content_recommendations(selected_movie, top_n)
     st.subheader(f"🎥 Movies similar to *{selected_movie}*:")
     st.table(results)
 # Trigger recommendation
 if selected_genres:
     result_df = recommend_movie(selected_genres, top_n)
     st.subheader("🎥 Recommended Movies:")
     st.table(result_df)
 else:
     st.info("👈 Please select at least one genre to see recommendations.")
