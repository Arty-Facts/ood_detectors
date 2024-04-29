import numpy as np
from sklearn.covariance import EmpiricalCovariance
import torch

class Residual():
    def __init__(self, dims=512, u=0):
        self.dims = dims
        self.u = u

    def fit(self, data, *args, **kwargs):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif isinstance(data, torch.utils.data.DataLoader):
            data = np.vstack([x.cpu().numpy() for x, _ in data])
        
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(data - self.u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        self.ns = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[self.dims:]]).T)
        return []

    def predict(self, data, *args, **kwargs):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif isinstance(data, torch.utils.data.DataLoader):
            data = np.vstack([x.cpu().numpy() for x, _ in data])
        
        return np.linalg.norm((data - self.u) @ self.ns, axis=-1)
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    
    def to(self, device):
        pass


    def state_dict(self):
        return {
            'dims': self.dims,
            'u': self.u,
            'ns': self.ns
        }
    
    def load_state_dict(self, state_dict):
        self.dims = state_dict['dims']
        self.u = state_dict['u']
        self.ns = state_dict['ns']
        return self
