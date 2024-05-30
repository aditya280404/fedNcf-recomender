import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense

def create_ncf_model(num_users, num_movies, latent_dim=8):
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))

    user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim)(user_input)
    movie_embedding = Embedding(input_dim=num_movies, output_dim=latent_dim)(movie_input)

    user_flat = Flatten()(user_embedding)
    movie_flat = Flatten()(movie_embedding)

    concat = Concatenate()([user_flat, movie_flat])

    dense_1 = Dense(64, activation='relu')(concat)
    dense_2 = Dense(32, activation='relu')(dense_1)

    output = Dense(1)(dense_2)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    return model


def load_weights(weights_path, model):
    with np.load(weights_path, allow_pickle=True) as data:
        loaded_weights = [data[key] for key in sorted(data.keys())]
        model.set_weights(loaded_weights)
