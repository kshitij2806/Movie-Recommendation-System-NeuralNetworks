"""
Recommendation Engine
Uses the trained RBM to generate movie recommendations for users.
"""

import os
import torch
import numpy as np
import pandas as pd

from src.rbm_model import RBM
from src.data_preprocessing import (
    load_ratings, load_movies, create_user_movie_matrix, binarize_ratings
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rbm_trained.pt")


class RecommendationEngine:
    """Generates movie recommendations using a trained RBM."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH

        # Load model
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.rbm = RBM(
            n_visible=checkpoint["n_visible"],
            n_hidden=checkpoint["n_hidden"],
        )
        self.rbm.load_state_dict(checkpoint["model_state"])
        self.rbm.eval()

        # Load data
        self.ratings_df = load_ratings()
        self.movies_df = load_movies()
        self.n_users = self.ratings_df["user_id"].max()
        self.n_movies = self.ratings_df["movie_id"].max()

        # Build full rating matrix
        self.full_matrix = create_user_movie_matrix(
            self.ratings_df, self.n_users, self.n_movies
        )

    def get_user_recommendations(self, user_id, top_n=10):
        """
        Get top-N movie recommendations for a given user.

        Args:
            user_id: 1-indexed user ID.
            top_n: Number of recommendations to return.

        Returns:
            List of dicts with movie_id, title, genres, predicted_score.
        """
        if user_id < 1 or user_id > self.n_users:
            return []

        # Get user's ratings
        user_ratings = self.full_matrix[user_id - 1].copy()
        rated_mask = user_ratings > 0

        # Binarize and normalize
        binary = np.zeros_like(user_ratings)
        binary[user_ratings >= 3] = 1
        binary[(user_ratings > 0) & (user_ratings < 3)] = -1
        normalized = (binary + 1) / 2  # to 0-1 range

        # Predict
        v = torch.FloatTensor(normalized).unsqueeze(0)
        mask = torch.FloatTensor(rated_mask.astype(float)).unsqueeze(0)
        predictions = self.rbm.predict(v, mask).squeeze().numpy()

        # Filter out already-rated movies and sort
        predictions[rated_mask] = -1  # exclude rated movies
        top_indices = np.argsort(predictions)[::-1][:top_n]

        recommendations = []
        for idx in top_indices:
            movie_id = idx + 1
            movie_info = self.movies_df[self.movies_df["movie_id"] == movie_id]
            if not movie_info.empty:
                row = movie_info.iloc[0]
                recommendations.append({
                    "movie_id": int(movie_id),
                    "title": row["title"],
                    "genres": row["genres"],
                    "score": round(float(predictions[idx]) * 5, 2),
                })

        return recommendations

    def get_similar_movies(self, movie_id, top_n=8):
        """
        Find movies similar to a given movie using RBM weight similarity.
        """
        if movie_id < 1 or movie_id > self.n_movies:
            return []

        # Use weight vectors as movie embeddings
        weights = self.rbm.W.data.numpy()
        movie_vec = weights[movie_id - 1]

        # Cosine similarity
        norms = np.linalg.norm(weights, axis=1)
        norms[norms == 0] = 1e-8
        similarities = np.dot(weights, movie_vec) / (norms * np.linalg.norm(movie_vec))
        similarities[movie_id - 1] = -1  # exclude self

        top_indices = np.argsort(similarities)[::-1][:top_n]

        results = []
        for idx in top_indices:
            mid = idx + 1
            movie_info = self.movies_df[self.movies_df["movie_id"] == mid]
            if not movie_info.empty:
                row = movie_info.iloc[0]
                results.append({
                    "movie_id": int(mid),
                    "title": row["title"],
                    "genres": row["genres"],
                    "similarity": round(float(similarities[idx]), 3),
                })

        return results

    def get_user_rated_movies(self, user_id):
        """Get movies a user has already rated."""
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        merged = user_ratings.merge(self.movies_df, on="movie_id")
        return merged[["movie_id", "title", "genres", "rating"]].to_dict("records")

    def get_popular_movies(self, top_n=20):
        """Get most popular movies by rating count and average."""
        stats = self.ratings_df.groupby("movie_id").agg(
            avg_rating=("rating", "mean"),
            n_ratings=("rating", "count")
        ).reset_index()

        # Weighted score: Bayesian average
        C = stats["n_ratings"].mean()
        m = stats["avg_rating"].mean()
        stats["score"] = (
            (stats["n_ratings"] * stats["avg_rating"] + C * m)
            / (stats["n_ratings"] + C)
        )

        top = stats.nlargest(top_n, "score")
        merged = top.merge(self.movies_df, on="movie_id")

        return merged[["movie_id", "title", "genres", "score", "n_ratings"]].to_dict("records")

    def get_stats(self):
        """Return dataset statistics."""
        return {
            "total_users": int(self.n_users),
            "total_movies": int(self.n_movies),
            "total_ratings": len(self.ratings_df),
            "avg_rating": round(float(self.ratings_df["rating"].mean()), 2),
            "model_params": sum(p.numel() for p in self.rbm.parameters()),
        }


# Singleton for the Flask app
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine
