import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine\_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# Must be at the top — avoids page config errors

st.set\_page\_config(page\_title="AI Movie Recommender", layout="centered")

# Sample Movie Data

movies = pd.DataFrame({
'title': \['Inception', 'Titanic', 'The Matrix', 'The Godfather', 'Avengers'],
'genres': \[\['Action', 'Sci-Fi'], \['Romance', 'Drama'], \['Sci-Fi', 'Action'], \['Crime', 'Drama'], \['Action', 'Fantasy']],
'rating': \[8.8, 7.8, 8.7, 9.2, 8.4]
})

# Genre list for UI

genre\_choices = sorted(set(genre for sublist in movies\['genres'] for genre in sublist))

def recommend\_movie(user\_genres, top\_n):
\# Transform genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre\_matrix = mlb.fit\_transform(movies\['genres'])

```
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
```

# UI Layout

st.title("🎬 AI-Powered Movie Recommender")
st.markdown("Select your favorite genres and get personalized movie suggestions.")

selected\_genres = st.multiselect("Choose your favorite genres:", genre\_choices)
top\_n = st.slider("Number of movie recommendations:", min\_value=1, max\_value=5, value=3)

# Trigger recommendation

if selected\_genres:
result\_df = recommend\_movie(selected\_genres, top\_n)
st.subheader("🎥 Recommended Movies:")
st.table(result\_df)
else:
st.info("👈 Please select at least one genre to see recommendations.")

