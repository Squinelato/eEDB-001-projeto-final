import torch.nn as nn
import torch

class MovieLensRecSys(nn.Module):
    '''
    Classe criada com o intuito de modelar a estrutura de torre-dupla,
    combinado com uma rede neural baseada em filtragem
    colaborativa por meio de redes neurais.
    '''
    def __init__(self, n_users, n_movies, embedding_size = 32):
        super().__init__()
        # definindo embedding para clientes, produtos e categorias
        self.users_embedding = nn.Embedding(n_users, embedding_size)
        self.movies_embedding = nn.Embedding(n_movies, embedding_size)
        # definindo primeira camada de reurônios totalmente conectados
        self.fully_conn_1 = nn.Linear(embedding_size * 2, 32)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.2)
        # # # definindo primeira camada de reurônios totalmente conectados
        self.fully_conn_2 = nn.Linear(32, 16)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=0.2)
        # definindo camada de saída como um neurônio
        self.output_layer = nn.Linear(16, 1)

    def forward(self, batch):
        # criando camada de entrada a partir de embeddings de clientes e produtos
        user_embeddings = self.users_embedding(batch['users'])
        movies_embeddings = self.movies_embedding(batch['movies'])
        # concatenando embeddings de usuários e livros
        concat_embeddings = torch.cat([user_embeddings, movies_embeddings], dim=1).to(torch.float32)
        # primeira camada totalmente conectada
        output = self.fully_conn_1(concat_embeddings)
        output = self.relu_1(output)
        output = self.dropout_1(output)
        # # # segunda camada totalmente conectada
        output = self.fully_conn_2(output)
        output = self.relu_2(output)
        output = self.dropout_2(output)
        # camada de saída
        output = self.output_layer(output)

        return output

class BasicRecSys(nn.Module):
    '''
    Classe criada com o intuito de modelar a estrutura de torre-dupla,
    isto é, um dos modelos clássicos de RecSys baseado em filtragem
    colaborativa por meio de redes neurais.
    '''
    def __init__(self, n_users, n_movies, embedding_size = 32):
        super().__init__()
        # definindo embedding para clientes, produtos e categorias
        self.users_embedding = nn.Embedding(n_users, embedding_size)
        self.movies_embedding = nn.Embedding(n_movies, embedding_size)
        # definindo produto dos embeddings
        self.dot = torch.matmul
        # definindo camada de saída como um neurônio

    def forward(self, batch):
        # criando camada de entrada a partir de embeddings de clientes e produtos
        user_embeddings = self.users_embedding(batch['users'])
        movies_embeddings = self.movies_embedding(batch['movies'])
        # realizando o produto dos vetores
        output = self.dot(user_embeddings, movies_embeddings.t())
        output = output.diagonal().unsqueeze(1)

        return output