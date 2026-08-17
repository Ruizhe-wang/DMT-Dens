from torch.utils.data import Sampler
import numpy as np

class MultiClusterBatchSampler(Sampler):
    def __init__(self, neighbors_index, clusters_per_batch=4, cluster_size=8, num_batches=100):
        self.neighbors_index = neighbors_index
        self.clusters_per_batch = clusters_per_batch
        self.cluster_size = cluster_size
        self.num_batches = num_batches
        
        # import pdb; pdb.set_trace()  # Debugging line to inspect the neighbors_index

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.clusters_per_batch):
                
                seed = np.random.randint(len(self.neighbors_index))
                cluster = self.neighbors_index[seed]
                cluster = cluster[cluster > -1]
                cluster = np.append(cluster, seed)
                if len(cluster) > self.cluster_size:
                    cluster = np.random.choice(cluster, self.cluster_size, replace=False)
                else:
                    cluster = np.random.choice(cluster, self.cluster_size, replace=True)
                
                batch.extend(cluster.tolist())
            yield batch

    def __len__(self):
        return self.num_batches