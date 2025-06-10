# For the purpose of simulating a collaborative filtering recommender system, the test set is used as a proxy for unseen user preferences.
# I rank the test items as if they were unseen, but they are actually held-out interactions.

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

# Load in data and join on movieId
ratings = pd.read_csv("./data/ml-latest-small/ratings.csv")
movies = pd.read_csv("./data/ml-latest-small/movies.csv")
ratings_with_titles = pd.merge(ratings, movies, on='movieId')

# Prepare data for Surprise library
reader = Reader(rating_scale=(0.5, 5.0)) # MovieLens uses 0.5 to 5.0 stars
data = Dataset.load_from_df(ratings_with_titles[['userId', 'movieId', 'rating']], reader)

# Train with Surprise's SVD. The input is going to be the triplet user, item, rating. rating is the target variable.
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset) # Train the model on known ratings

# Predict ratings for user-item pairs in the test set.
# This is the **candidate generation** phase, so the model generates a pool of candidate movies for each user by estimating how likely the user will like them
predictions = model.test(testset) # Functionally equivalent to .predict() for scikit learn models

# Get top-N recommendations per user
# This is the **ranking** phase, so I rank all candidate movies by their predicted ratings then keep only the top N movies with the highest predicted ratings.
# The output is a dictionary mapping each userId to their top N recommended movies with estimated scores
def get_top_n(predictions, n=5):
    top_n = defaultdict(list)

    # Collect all predictions for each user (candidate movies + predicted ratings)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sort the candidate movies for each user by predicted rating in descending order, then retain only the top N highest ranked movies
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x:x[1], reverse=True) # Sort by predicted rating (ranking)
        top_n[uid] = user_ratings[:n] # Keep top N ranked candidates
    return top_n
# Generate top 5 movie recommendations per user. In other words, rank the candidate pool.
top_n = get_top_n(predictions, n=5)

# Convert recommendations to dataframe for Tableau and export.
rec_list = []
for uid, movie_ratings in top_n.items():
    for iid, est_rating in movie_ratings:
        title = movies[movies['movieId'] == iid]['title'].values[0]
        rec_list.append([uid, iid, title, est_rating])
recommendations_df = pd.DataFrame(rec_list, columns=['userId', 'movieId', 'title', 'predicted_rating'])
recommendations_df.to_csv('./data/top_recommendations.csv')

# Optional output: Export full user-movie ratings with titles
full_ratings_df = ratings_with_titles[['userId', 'movieId', 'title', 'rating']]
full_ratings_df.to_csv('./data/full_user_ratings.csv', index=False)