# Install needed packages first:
# pip install pandas scikit-learn scikit-surprise

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load datasets
movies = pd.read_csv('movies.csv')   # movieId, title, genres
ratings = pd.read_csv('ratings.csv') # userId, movieId, rating

# Collaborative filtering with Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

# Get top N recommendations for a user
def get_collab_recommendations(user_id, n=5):
    movie_ids = ratings['movieId'].unique()
    predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_movies = [p.iid for p in predictions[:n]]
    return movies[movies['movieId'].isin(top_movies)][['title', 'genres']]

# Content-based filtering (TF-IDF on genres)
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_content_recommendations(title, n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]

# Example usage
print("Collaborative recommendations for user 1:")
print(get_collab_recommendations(user_id=1))

print("\nContent-based recommendations similar to 'Toy Story':")
print(get_content_recommendations(title='Toy Story'))
