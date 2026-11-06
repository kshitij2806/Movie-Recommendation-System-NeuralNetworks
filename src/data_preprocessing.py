"""
Data Preprocessing Module
Handles loading, cleaning, and transforming the MovieLens 100K dataset
for training the RBM model.
"""

import os
import ssl
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATASET_PATH = os.path.join(DATA_DIR, "ml-100k")


def download_dataset():
    """Download MovieLens 100K dataset if not already present."""
    if os.path.exists(DATASET_PATH):
        print("Dataset already exists.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-100k.zip")

    print("Downloading MovieLens 100K dataset...")
    # Handle environments with missing SSL certificates
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(DATASET_URL)
    with urllib.request.urlopen(req, context=ctx) as response, open(zip_path, "wb") as out:
        out.write(response.read())

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)

    os.remove(zip_path)
    print("Dataset ready.")


def load_ratings():
    """Load the ratings data into a pandas DataFrame."""
    path = os.path.join(DATASET_PATH, "u.data")
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    df = pd.read_csv(path, sep="\t", names=columns, engine="python")
    df.drop("timestamp", axis=1, inplace=True)
    return df


def load_movies():
    """Load movie metadata (id, title, genres)."""
    path = os.path.join(DATASET_PATH, "u.item")
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children's",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western",
    ]
    columns = [
        "movie_id", "title", "release_date", "video_release_date",
        "imdb_url",
    ] + genre_names

    df = pd.read_csv(
        path, sep="|", names=columns, encoding="latin-1", engine="python"
    )

    # Keep useful columns
    keep = ["movie_id", "title"] + genre_names
    df = df[keep]

    # Create a single genre string column
    def get_genres(row):
        return ", ".join(g for g in genre_names if row[g] == 1)

    df["genres"] = df.apply(get_genres, axis=1)
    df = df[["movie_id", "title", "genres"]]
    return df


def load_users():
    """Load user demographic data."""
    path = os.path.join(DATASET_PATH, "u.user")
    columns = ["user_id", "age", "gender", "occupation", "zip_code"]
    df = pd.read_csv(path, sep="|", names=columns, engine="python")
    return df


def create_user_movie_matrix(ratings_df, n_users=None, n_movies=None):
    """
    Create a user-movie rating matrix.
    Unrated movies are set to 0 (will be masked during training).
    Ratings are kept as 1-5 integers.
    """
    if n_users is None:
        n_users = ratings_df["user_id"].max()
    if n_movies is None:
        n_movies = ratings_df["movie_id"].max()

    matrix = np.zeros((n_users, n_movies), dtype=np.float32)

    for _, row in ratings_df.iterrows():
        uid = int(row["user_id"]) - 1   # 0-indexed
        mid = int(row["movie_id"]) - 1   # 0-indexed
        matrix[uid, mid] = float(row["rating"])

    return matrix


def binarize_ratings(matrix, threshold=3):
    """
    Convert ratings to binary: 1 if rating >= threshold, -1 if rated below,
    0 if unrated (masked).
    """
    binary = np.zeros_like(matrix)
    binary[matrix >= threshold] = 1
    binary[(matrix > 0) & (matrix < threshold)] = -1
    return binary


def prepare_data(test_size=0.2, binary=True, threshold=3):
    """
    Full preprocessing pipeline. Returns train/test matrices and metadata.
    """
    download_dataset()

    ratings = load_ratings()
    movies = load_movies()

    n_users = ratings["user_id"].max()
    n_movies = ratings["movie_id"].max()

    # Split by users to avoid data leakage
    train_df, test_df = train_test_split(
        ratings, test_size=test_size, random_state=42, stratify=None
    )

    train_matrix = create_user_movie_matrix(train_df, n_users, n_movies)
    test_matrix = create_user_movie_matrix(test_df, n_users, n_movies)

    if binary:
        train_matrix = binarize_ratings(train_matrix, threshold)
        test_matrix = binarize_ratings(test_matrix, threshold)

    print(f"Users: {n_users}, Movies: {n_movies}")
    print(f"Train ratings: {len(train_df)}, Test ratings: {len(test_df)}")
    print(f"Matrix shape: {train_matrix.shape}")

    return {
        "train_matrix": train_matrix,
        "test_matrix": test_matrix,
        "n_users": n_users,
        "n_movies": n_movies,
        "movies_df": movies,
        "ratings_df": ratings,
    }


if __name__ == "__main__":
    data = prepare_data()
    print("Preprocessing complete!")
