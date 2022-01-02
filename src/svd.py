import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy


def create_user_item_matrix(ratings) -> pd.DataFrame:
    mat = ratings.pivot(index="userId", columns="movieId", values="rating")
    # 유저가 평점을 매긴 영화는 '본 영화'로 간주하고 1로 지정합니다
    mat[~mat.isna()] = 1
    # 유저가 평점을 매기지 않은 영화는 NaN인데
    # '보지 않은 영화'로 간주하고 0으로 바꿔줍니다
    mat.fillna(0, inplace=True)
    return mat


def get_svd_prediction(user_item_matrix, k):
    # U, sigma, V 를 얻어옵니다.
    u, s, vh = scipy.sparse.linalg.svds(user_item_matrix.to_numpy(), k=k)
    # 이들을 다시 곱해서 원본 행렬을 복원해봅니다
    # sigma는 주대각선 성분만 반환되므로 np.diag 함수로 대각행렬로 바꿔줍니다
    preds = np.dot(np.dot(u, np.diag(s)), vh)

    # 결과를 DataFrame으로 만들고, 0~1 사이의 값으로 정규화합니다.
    preds = pd.DataFrame(preds, columns=user_item_matrix.columns, index=user_item_matrix.index)
    preds = (preds - preds.min()) / (preds.max() - preds.min())
    return preds


class SVD:
    def __init__(self, ratings, movies, k): 
        user_item_matrix = create_user_item_matrix(ratings)
        self.preds = get_svd_prediction(user_item_matrix, k)
        self.ratings = ratings
        self.movies = movies

    def get_recommendations(self, user_id, top_k=None):
        user_movie_ids = self.ratings[self.ratings.userId == user_id].movieId
        user_movies = self.movies[self.movies.movieId.isin(user_movie_ids)]

        # 복원된 행렬에서 유저 row만 가져온 뒤 내림차순으로 정렬합니다
        user_predictions = self.preds.loc[user_id].sort_values(ascending=False)
        # 이미 유저가 본 영화는 제외합니다
        user_predictions = user_predictions[~user_predictions.index.isin(user_movie_ids)]
        # 10개 영화의 정보를 가져옵니다
        user_recommendations = self.movies[self.movies.movieId.isin(user_predictions.index)]
        user_recommendations["recommendation_score"] = user_predictions.values
        
        return user_recommendations if top_k is None else user_recommendations.head(top_k)