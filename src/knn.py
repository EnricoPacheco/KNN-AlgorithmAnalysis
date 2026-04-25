#retirado do repositório https://github.com/rushter/MLAlgorithms/tree/master/mla

import numpy as np
from collections import Counter
from scipy.spatial.distance import euclidean

class KNNBase:
    def __init__(self, k=5, distance_func=euclidean):
        """
        Base para o classificador kNN.
        k: número de vizinhos[cite: 16].
        distance_func: função de distância (Euclidiana por padrão).
        """
        self.k = None if k == 0 else k
        self.distance_func = distance_func

    def fit(self, X, y):
        # O kNN armazena os dados de treino para consulta posterior
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X):
        predictions = [self._predict_x(x) for x in X]
        return np.array(predictions)

    def _predict_x(self, x):
        # 1. Calcula distâncias entre x e todos os exemplos de treino
        distances = (self.distance_func(x, example) for example in self.X)

        # 2. Ordena exemplos pela distância e mantém o valor do alvo (target)
        neighbors = sorted(
            ((dist, target) for (dist, target) in zip(distances, self.y)),
            key=lambda item: item[0],
        )

        # 3. Pega os alvos dos k vizinhos mais próximos
        neighbors_targets = [target for (_, target) in neighbors[: self.k]]

        return self.aggregate(neighbors_targets)

    def aggregate(self, neighbors_targets):
        raise NotImplementedError()

class KNNClassifier(KNNBase):
    def aggregate(self, neighbors_targets):
        """Retorna a label mais comum (votação majoritária)."""
        most_common_label = Counter(neighbors_targets).most_common(1)[0][0]
        return most_common_label