import pandas as pd
from tensorflow.keras.utils import get_file
import zipfile

def load_movielens_data():
    movielens_data = get_file("ml-latest-small.zip", "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip", extract=True)
    movielens_dir = movielens_data.replace(".zip", "")
    with zipfile.ZipFile(movielens_data, "r") as zip_ref:
        zip_ref.extractall(movielens_dir)
    ratings = pd.read_csv(movielens_dir + '/ml-latest-small/ratings.csv')
    num_users = ratings['userId'].nunique()
    num_movies = ratings['movieId'].nunique()
    return ratings, num_users, num_movies