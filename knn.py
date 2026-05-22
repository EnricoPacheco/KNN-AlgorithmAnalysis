import numpy as np
from scipy.spatial.distance import euclidean

class KNNBase:
    """
    Classe base para o algoritmo k-Nearest Neighbors (kNN).
    Implementa a lógica central de busca de vizinhos.
    """
    def __init__(self, k=5, distance_func=euclidean):
        """
        Inicializa o classificador base.
        
        Args:
            k (int): Número de vizinhos a serem considerados.
            distance_func (callable): Função para cálculo de distância.
        """
        self.k = None if k == 0 else k
        self.distance_func = distance_func

    def fit(self, X, y, sample_weights=None):
        """
        Treina o modelo armazenando os dados em memória (Lazy Learning).
        
        Args:
            X (array-like): Matriz de features de treinamento.
            y (array-like): Vetor de rótulos alvo.
            sample_weights (array-like, opcional): Pesos associados a cada amostra.
        """
        self.X = np.array(X)
        self.y = np.array(y)

        # Inicializa pesos unitários caso não sejam fornecidos
        if sample_weights is None:
            self.weights = np.ones(len(y))
        else:
            self.weights = np.array(sample_weights)

    def predict(self, X):
        """
        Gera previsões para um conjunto de amostras.
        
        Args:
            X (array-like): Matriz de features para previsão.
            
        Returns:
            np.array: Vetor de rótulos previstos.
        """
        predictions = [self._predict_x(x) for x in X]
        return np.array(predictions)

    def _predict_x(self, x):
        """
        Realiza a busca dos vizinhos mais próximos para uma única amostra.
        
        Args:
            x (array-like): Amostra individual.
            
        Returns:
            O rótulo previsto agregado pela função da subclasse.
        """
        # 1. Calcula a distância entre 'x' e todos os pontos de treino
        distances = (self.distance_func(x, example) for example in self.X)

        # 2. Ordena os vizinhos pela menor distância mantendo rótulos e pesos
        neighbors = sorted(
            ((dist, target, weight) for (dist, target, weight) in zip(distances, self.y, self.weights)),
            key=lambda item: item[0],
        )

        # 3. Extrai apenas os k vizinhos mais próximos
        neighbors_targets_weights = [(target, weight) for (_, target, weight) in neighbors[: self.k]]

        # 4. Agrega a decisão baseada na implementação específica
        return self.aggregate(neighbors_targets_weights)

    def aggregate(self, neighbors_targets_weights):
        """Método abstrato de agregação de votos."""
        raise NotImplementedError()

class KNNClassifier(KNNBase):
    """
    Classificador kNN padrão.
    Resolve previsões através de votação majoritária ou ponderada.
    """
    def aggregate(self, neighbors_targets_weights):
        """
        Agrega os votos dos vizinhos somando seus pesos.
        No kNN padrão sem pesos, cada vizinho contribui com +1.
        """
        class_votes = {}
        for target, weight in neighbors_targets_weights:
            class_votes[target] = class_votes.get(target, 0) + weight

        # Retorna a classe com a maior densidade de votos
        most_common_label = max(class_votes, key=class_votes.get)
        return most_common_label

class AdaptiveKNNClassifier(KNNBase):
    def __init__(self, k_min=3, k_max=11, distance_func=euclidean):
        """
        Inicializa o modelo adaptativo definindo os limites de vizinhança.
        
        Args:
            k_min (int): k utilizado em regiões densas (livres de ruído).
            k_max (int): k utilizado em regiões esparsas (vulneráveis a outliers).
        """
        super().__init__(k=k_max, distance_func=distance_func)
        self.k_min = k_min
        self.k_max = k_max

    def fit(self, X, y, sample_weights=None):
        """
        Treina o modelo e calibra o limiar dinâmico de tolerância a ruído.
        
        Args:
            sample_weights (array-like): Scores LOF de cada amostra.
        """
        super().fit(X, y, sample_weights)
        if sample_weights is not None:
            # O limiar dinâmico define o ponto de corte: a média dos 25% piores scores
            # serve como fronteira entre uma região 'limpa' e uma 'ruidosa'.
            self.noise_threshold = np.percentile(sample_weights, 25)
        else:
            self.noise_threshold = -1.5

    def _predict_x(self, x):
        """
        Efetua a previsão adaptando 'k' ao contexto topológico da amostra.
        """
        distances = (self.distance_func(x, example) for example in self.X)

        # Ordena mantendo o score LOF do vizinho (injetado via sample_weights)
        neighbors = sorted(
            ((dist, target, lof_score) for (dist, target, lof_score) in zip(distances, self.y, self.weights)),
            key=lambda item: item[0],
        )

        # Amostra a vizinhança máxima permitida
        top_k_max = neighbors[:self.k_max]
        
        # Estima a integridade da vizinhança baseada na média dos scores LOF
        avg_lof = np.mean([lof_score for _, _, lof_score in top_k_max])

        # Se a região for topologicamente segura (score acima do limiar crítico)
        # prioriza-se a precisão geométrica reduzindo-se o raio de busca (k_min)
        if avg_lof > self.noise_threshold:
            chosen_neighbors = top_k_max[:self.k_min]
        # Caso contrário, mantém-se o k_max para diluir o ruído em um consenso maior
        else:
            chosen_neighbors = top_k_max

        # Formatação de retorno (descarta o score LOF para a contagem final)
        formatted_neighbors = [(target, 1) for (_, target, _) in chosen_neighbors]
        return self.aggregate(formatted_neighbors)

    def aggregate(self, neighbors_targets_weights):
        class_votes = {}
        for target, _ in neighbors_targets_weights:
            class_votes[target] = class_votes.get(target, 0) + 1

        return max(class_votes, key=class_votes.get)
from scipy.spatial.distance import cdist

def compute_local_outlier_factor(X, k=20):
    n_samples = len(X)
    if n_samples <= k:
        k = n_samples - 1
        
    if k <= 0:
        return np.zeros(n_samples)

    dist_matrix = cdist(X, X, metric='euclidean')

    k_distances = np.zeros(n_samples)
    neighborhoods = []
    for i in range(n_samples):
        sorted_indices = np.argsort(dist_matrix[i])
        neighbors = sorted_indices[1:k+1]
        neighborhoods.append(neighbors)
        k_distances[i] = dist_matrix[i, neighbors[-1]]

    lrds = np.zeros(n_samples)
    for i in range(n_samples):
        reach_dist_sum = 0
        for j in neighborhoods[i]:
            reach_dist = max(k_distances[j], dist_matrix[i, j])
            reach_dist_sum += reach_dist
        if reach_dist_sum == 0:
            lrds[i] = 1e10
        else:
            lrds[i] = len(neighborhoods[i]) / reach_dist_sum

    lofs = np.zeros(n_samples)
    for i in range(n_samples):
        lrd_sum = sum(lrds[j] for j in neighborhoods[i])
        lof = (lrd_sum / lrds[i]) / len(neighborhoods[i])
        lofs[i] = -lof

    return lofs
