import numpy as np
from scipy.spatial.distance import euclidean

class KNNBase:
    def __init__(self, k=5, distance_func=euclidean):
        """
        Base para o classificador kNN.
        k: número de vizinhos.
        distance_func: função de distância (Euclidiana por padrão).
        """
        self.k = None if k == 0 else k
        self.distance_func = distance_func

    def fit(self, X, y, sample_weights=None):
        # Armazena os dados de treino
        self.X = np.array(X)
        self.y = np.array(y)

        # Se não passarmos pesos, todos os exemplos têm peso 1 (kNN tradicional)
        if sample_weights is None:
            self.weights = np.ones(len(y))
        else:
            self.weights = np.array(sample_weights)

    def predict(self, X):
        predictions = [self._predict_x(x) for x in X]
        return np.array(predictions)

    def _predict_x(self, x):
        # 1. Calcula distâncias entre x e todos os exemplos de treino
        distances = (self.distance_func(x, example) for example in self.X)

        # 2. Ordena exemplos pela distância e mantém o valor do alvo e o peso
        neighbors = sorted(
            ((dist, target, weight) for (dist, target, weight) in zip(distances, self.y, self.weights)),
            key=lambda item: item[0],
        )

        # 3. Pega os alvos e os pesos dos k vizinhos mais próximos
        neighbors_targets_weights = [(target, weight) for (_, target, weight) in neighbors[: self.k]]

        return self.aggregate(neighbors_targets_weights)

    def aggregate(self, neighbors_targets_weights):
        raise NotImplementedError()

class KNNClassifier(KNNBase):
    def aggregate(self, neighbors_targets_weights):
        """Votação Ponderada: soma os pesos para cada classe e retorna a maior."""
        class_votes = {}
        for target, weight in neighbors_targets_weights:
            class_votes[target] = class_votes.get(target, 0) + weight

        # Retorna a classe com a maior soma de pesos
        most_common_label = max(class_votes, key=class_votes.get)
        return most_common_label