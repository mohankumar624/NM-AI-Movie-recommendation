# Install needed packages first:

# pip install pandas scikit-learn surprise

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model\_selection import train\_test\_split
from sklearn.feature\_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear\_kernel

# Load datasets

movies = pd.read\_csv('movies.csv')   # movieId, title, genres
ratings = pd.read\_csv('ratings.csv') # userId, movieId, rating

# Collaborative filtering with Surprise

reader = Reader(rating\_scale=(0.5, 5.0))
data = Dataset.load\_from\_df(ratings\[\['userId', 'movieId', 'rating']], reader)
trainset, testset = train\_test\_split(data, test\_size=0.2)

model = SVD()
model.fit(trainset)

# Get top N recommendations for a user

def get\_collab\_recommendations(user\_id, n=5):
movie\_ids = ratings\['movieId'].unique()
predictions = \[model.predict(user\_id, movie\_id) for movie\_id in movie\_ids]
predictions.sort(key=lambda x: x.est, reverse=True)
top\_movies = \[p.iid for p in predictions\[:n]]
return movies\[movies\['movieId'].isin(top\_movies)]\[\['title', 'genres']]

# Content-based filtering (TF-IDF on genres)

tfidf = TfidfVectorizer(stop\_words='english')
movies\['genres'] = movies\['genres'].fillna('')
tfidf\_matrix = tfidf.fit\_transform(movies\['genres'])
cosine\_sim = linear\_kernel(tfidf\_matrix, tfidf\_matrix)

indices = pd.Series(movies.index, index=movies\['title']).drop\_duplicates()

def get\_content\_recommendations(title, n=5):
idx = indices\[title]
sim\_scores = list(enumerate(cosine\_sim\[idx]))
sim\_scores = sorted(sim\_scores, key=lambda x: x\[1], reverse=True)
sim\_scores = sim\_scores\[1\:n+1]
movie\_indices = \[i\[0] for i in sim\_scores]
return movies.iloc\[movie\_indices]\[\['title', 'genres']]

# Example usage

print("Collaborative recommendations for user 1:")
print(get\_collab\_recommendations(user\_id=1))

print("\nContent-based recommendations similar to 'Toy Story':")
print(get\_content\_recommendations(title='Toy Story'))

ModuleNotFoundError                       Traceback (most recent call last) <ipython-input-3-b82aa3f8b80a> in \<cell line: 0>()
3
4 import pandas as pd
\----> 5 from surprise import Dataset, Reader, SVD
6 from surprise.model\_selection import train\_test\_split
7 from sklearn.feature\_extraction.text import TfidfVectorizer

ModuleNotFoundError: No module named 'surprise'

---

NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To view examples of installing some common dependencies, click the
"Open Examples" button below.
