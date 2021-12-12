import pandas as pd
from sklearn.neighbors import NearestNeighbors

class ClassicRecommender:
    def __init__(self, data):
        self.data = data
        
    def get_most_rated(self, top_k=100):
        x = self.data.sort_values(["ratings_count", "ratings_mean"], ascending=[False, False])
        return x.head(top_k)
    
    def get_top_rated(self, top_k=100, min_ratings_count=50):
        x = self.data.sort_values(["ratings_mean", "ratings_count"], ascending=[False, False])
        x = x[x.ratings_count >= min_ratings_count]
        return x.head(top_k)
    
class ContentBasedRecommender:
    
    def __init__(self, data, max_neighbors):
        self.data = data
        self.genre_columns = data.columns[1:25]
        self.nn = NearestNeighbors(max_neighbors)
        self.nn.fit(self.data[self.genre_columns])
        
    def recommend_by_genre(self, movie_id):
        movie = self.data.loc[[movie_id], self.genre_columns]
        dists, ids = self.nn.kneighbors(movie)
        
        dists = pd.Series(dists[0], index=self.data.index[ids[0]], name="distance")
        movies = self.data.iloc[ids[0], :]
        movies = pd.concat([movies, dists], axis=1)
        movies = movies.sort_values(by=["distance", "ratings_mean", "ratings_count"], ascending=[True,False,False])
        return movies
    