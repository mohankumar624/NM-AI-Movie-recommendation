import pandas as pd
import numpy as np
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# Sample Movie Data (can be replaced with TMDB or MovieLens data)
movies = pd.DataFrame({
    'title': ['Inception', 'Titanic', 'The Matrix', 'The Godfather', 'Avengers'],
    'genres': [['Action', 'Sci-Fi'], ['Romance', 'Drama'], ['Sci-Fi', 'Action'], ['Crime', 'Drama'], ['Action', 'Fantasy']],
    'rating': [8.8, 7.8, 8.7, 9.2, 8.4]
})

# Prepare genre vectors
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genres'])

# Build content-based similarity matrix
def recommend_movie(user_genres, top_n=3):
    if not user_genres:
        return "Please select at least one genre."
    
    # Convert input genres to vector
    user_vector = mlb.transform([user_genres])
    scores = cosine_similarity(user_vector, genre_matrix)[0]
    
    # Recommend top N movies
    top_indices = np.argsort(scores)[::-1][:top_n]
    recommended = movies.iloc[top_indices][['title', 'genres', 'rating']]
    recommended['genres'] = recommended['genres'].apply(lambda g: ', '.join(g))
    return recommended.to_markdown(index=False)

# Create genre list for the checkbox
genre_choices = sorted(set(g for sublist in movies['genres'] for g in sublist))

# Define Gradio Blocks interface
with gr.Blocks() as demo:
    with gr.Row():
        genre_input = gr.CheckboxGroup(choices=genre_choices, label="Choose your favorite genres")
        num_recommendations = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Number of Recommendations")
    
    output = gr.Markdown()
    
    genre_input.change(recommend_movie, inputs=[genre_input, num_recommendations], outputs=output)
    num_recommendations.change(recommend_movie, inputs=[genre_input, num_recommendations], outputs=output)

demo.launch()
