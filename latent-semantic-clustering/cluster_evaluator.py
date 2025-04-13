import pandas as pd

class ClusterEvaluator:
    def __init__(self, pred_clusters, true_labels):
        self.df = pd.DataFrame({'Cluster': pred_clusters, 'Label': true_labels})
        self.mapping = self.df.groupby('Cluster')['Label'].agg(lambda x: x.mode()[0]).to_dict()
        
    @property
    def accuracy(self):
        predicted = self.df['Cluster'].map(self.mapping)
        return (predicted == self.df['Label']).mean(), self.mapping
