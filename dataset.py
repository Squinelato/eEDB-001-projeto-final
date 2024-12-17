from sklearn import preprocessing

import torch
import pandas as pd
import numpy as np

class MovieLens(torch.utils.data.Dataset):
    '''
    Classe criada com o intuito de ajustar o dataset pandas ao
    treinamento de modelos utilizando o PyTorch, especialmente do que se
    diz respeito à utilização de lotes (batches) durante o treinamento.
    '''
    def __init__(self, dataset_path: str, device: torch.device):
        """
        Construtor da classe, responsável por ler os dados e organizar os dados
        """
        self.device = device

        dataset = pd.read_csv(dataset_path, decimal='.')

        self.user_encoder = preprocessing.LabelEncoder()
        self.users = self.user_encoder.fit_transform(dataset['userId'].values)

        self.movie_encoder = preprocessing.LabelEncoder()
        self.movies = self.movie_encoder.fit_transform(dataset['movieId'].values)
        self.ratings = dataset['rating'].values

        self.n_unique_users = len(np.unique(self.users))
        self.n_unique_movies = len(np.unique(self.movies))

    def __len__(self) -> int:
        """
        Retorna o número de avaliações do conjunto de dados
        """
        return self.ratings.shape[0]

    def __getitem__(self, item) -> dict[torch.tensor]:
        """
        Retorna itens do conjunto de dados em lotes
        """
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]

        return {
            "users": torch.tensor(users, device=self.device, dtype=torch.long),
            "movies": torch.tensor(movies, device=self.device, dtype=torch.long),
            "ratings": torch.tensor(ratings, device=self.device, dtype=torch.long),
        }

    def get_original_user_id(self, users):
        """
        Retorna o ID original do usuário
        """
        return self.user_encoder.inverse_transform(users)
    
    def get_original_movie_id(self, movies):
        """
        Retorna o ID original do filme
        """
        return self.movie_encoder.inverse_transform(movies)
