# Collaborative Filtering Recommender System with SVD

This project implements a collaborative filtering movie recommender system using the [Surprise](http://surpriselib.com/) library's SVD (Singular Value Decomposition) algorithm. It trains on user-movie ratings from the [MovieLens dataset](https://grouplens.org/datasets/movielens/latest/) and outputs personalized recommendations for each user.

The final recommendations are visualized in Tableau:
[View Dashboard on Tableau Public](https://public.tableau.com/app/profile/morgan.dean8491/viz/collaborative-filtering/Dashboard1)

## Dataset

This project uses the MovieLens Latest Small dataset, which contains:

- `ratings.csv`: userId, movieId, rating, timestamp
- `movies.csv`: movieId, title, genres

Dataset source: https://grouplens.org/datasets/movielens/latest/