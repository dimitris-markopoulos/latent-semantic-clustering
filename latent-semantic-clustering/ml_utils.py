import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

class ClusterEvaluator:
    def __init__(self, pred_clusters, true_labels):
        self.df = pd.DataFrame({'Cluster': pred_clusters, 'Label': true_labels})
        self.mapping = self.df.groupby('Cluster')['Label'].agg(lambda x: x.mode()[0]).to_dict()
        
    @property
    def accuracy(self):
        predicted = self.df['Cluster'].map(self.mapping)
        return (predicted == self.df['Label']).mean(), self.mapping

class SilhouetteEvaluator:
    def __init__(self, X, clusterer, k_range=range(2, 11)):
        self.X = X
        self.clusterer = clusterer
        self.k_range = k_range
        self.scores = {}
        self.best_k = None

    def evaluate(self):
        for k in self.k_range:
            model = self.clusterer(n_clusters=k)
            labels = model.fit_predict(self.X)
            score = silhouette_score(self.X, labels)
            self.scores[k] = score
        self.best_k = max(self.scores, key=self.scores.get)
        return self.best_k, self.scores[self.best_k]

    def plot(self):
        if not self.scores:
            raise ValueError("Run evaluate() first.")
        plt.figure(figsize=(8, 5))
        plt.plot(list(self.scores.keys()), list(self.scores.values()), marker='o', linestyle='-', color='b', label='Silhouette Score')
        plt.axvline(self.best_k, color='r', linestyle='--', label=f'Best K = {self.best_k}')
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Average Silhouette Score")
        plt.title("Silhouette Score for Different K Values")
        plt.legend()
        plt.tight_layout()
        plt.show()
