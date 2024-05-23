import torch
import ood_detectors.likelihood as likelihood
import numpy as np

class RDM():
    def __init__(self, feat_dim, number_of_RDMs=2):
        super().__init__()
        self.feat_dim = feat_dim
        self.number_of_RDMs = number_of_RDMs
        self.ood_detectors = [
            likelihood.subVPSDE_RDM(feat_dim) for _ in range(number_of_RDMs)
        ]
        self.name = f"RDM_{self.ood_detectors[0].name}x{number_of_RDMs}"
        self.device = "cpu"

    def to(self, device):
        for ood_detector in self.ood_detectors:
            ood_detector.to(device)
        self.device = device
        return self

    def load_state_dict(self, state_dict):
        for ood_detector, state_dict_ in zip(self.ood_detectors, state_dict):
            ood_detector.load_state_dict(state_dict_)

    def state_dict(self):
        return [ood_detector.state_dict() for ood_detector in self.ood_detectors]
    
    def fit(self, dataset, n_epochs, batch_size, num_workers=0, update_fn=None, verbose=True, collate_fn=None):
        losses = []
        for ood_detector in self.ood_detectors:
            loss = ood_detector.fit(dataset, n_epochs, batch_size, num_workers, update_fn, verbose, collate_fn)
            losses.append(loss)
        return losses
    
    def predict(self, x, *args, reduce=True, **kwargs):
        if reduce:
            return np.stack([ood_detector.predict(x,*args, **kwargs) for ood_detector in self.ood_detectors]).mean(axis=0)
        else:
            return np.stack([ood_detector.predict(x,*args, **kwargs) for ood_detector in self.ood_detectors])
    


        
    
        