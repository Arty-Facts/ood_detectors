import faiss
import numpy as np
import torch

class KNN:
    def __init__(
                self,
                k: int = 1,
                eps: float = 1e-8,
            ):
        self.k = k  
        self.eps = eps
        
    def normalizer(self, x: np.ndarray):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + self.eps)
    
    def fit(self, data, *args, collate_fn=None, **kwargs):
        """
        Fit the KNN model to the given data.

        Args:
            data (array-like or torch.Tensor or torch.utils.data.DataLoader): Input data for fitting the model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list: An empty list.

        """
            
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif isinstance(data, torch.utils.data.Dataset):
            if collate_fn is None and getattr(data, 'collate_fn', None) is not None:
                collate_fn = data.collate_fn
            if collate_fn is None:
                data = np.vstack([x.cpu().numpy() for x, *_ in data])
            else:
                data = np.vstack([collate_fn([d]).cpu().numpy() for d in data])
                
        self.index = faiss.IndexFlatL2(data.shape[1])
        self.index.add(self.normalizer(data))
        
        return [-1]
    
    def predict(self, data, batch_size=1024, *args, collate_fn=None, **kwargs):
        """
        Predict the outlier scores for the given data in batches.

        Args:
            data (array-like, torch.Tensor, torch.utils.data.DataLoader, or torch.utils.data.Dataset): Input data for predicting the outlier scores.
            batch_size (int): The size of the batches to use for prediction.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
        """
        def to_numpy(data_batch):
            if isinstance(data_batch, torch.Tensor):
                return data_batch.cpu().numpy()
            elif isinstance(data_batch, (list, tuple)):
                return np.array(data_batch)
            return data_batch

        if isinstance(data, (list, tuple, np.ndarray)):
            data = torch.tensor(data, dtype=torch.float32)
            dataset = torch.utils.data.TensorDataset(data)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        elif isinstance(data, torch.Tensor):
            dataset = torch.utils.data.TensorDataset(data)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        elif isinstance(data, torch.utils.data.Dataset):
            if collate_fn is None and getattr(data, 'collate_fn', None) is not None:
                collate_fn = data.collate_fn
            data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
        elif isinstance(data, torch.utils.data.DataLoader):
            data_loader = data
        else:
            raise TypeError("Unsupported data type: {}".format(type(data)))
        
        distances = []
        
        for batch in data_loader:
            batch_numpy = to_numpy(batch)
            batch_numpy = batch_numpy.astype(np.float32)
            D, _ = self.index.search(self.normalizer(batch_numpy), self.k)
            distances.append(D)
            
        distances = np.concatenate(distances)
        
        return distances.mean(axis=-1)