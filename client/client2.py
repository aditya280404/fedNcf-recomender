import flwr as fl
import tensorflow as tf
from NCFMovieLens.model import create_ncf_model
from NCFMovieLens.data import load_movielens_data

# Load the data
ratings, num_users, num_movies = load_movielens_data()

# Define the Flower client
class Client(fl.client.NumPyClient):
    def __init__(self):
        self.model = create_ncf_model(num_users, num_movies)

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        # Extract data for training
        user_ids = ratings["userId"].values
        movie_ids = ratings["movieId"].values
        labels = ratings["rating"].values
        # Train the model
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit([user_ids, movie_ids], labels, epochs=1, batch_size=32, verbose=1)
        return self.model.get_weights(), len(ratings), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # Extract data for evaluation
        user_ids = ratings["userId"].values
        movie_ids = ratings["movieId"].values
        labels = ratings["rating"].values
        # Evaluate the model
        loss = self.model.evaluate([user_ids, movie_ids], labels)
        return loss, len(ratings), {}

# Start the Flower client
if __name__ == "__main__":
    fl.client.start_client(server_address="localhost:8080", client=Client().to_client())
