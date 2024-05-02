import torch
from ood_detectors import train, losses, ood_utils
from ood_detectors.models import SimpleMLP
from ood_detectors.sde import VESDE, subVPSDE, VPSDE
import torch.optim as optim
import functools


class Likelihood:
    """Base class for likelihood function implementations for different SDEs with dependency injection."""

    def __init__(self, sde, 
                 optimizer=functools.partial(
                    optim.Adam,
                    lr=0.0002,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0,
                 ), 
                 model=None, 
                 feat_dim=None,
                 ):
        self.model = model
        if model is None:
            assert feat_dim is not None, "feat_dim must be provided if model is None"
            self.model = SimpleMLP(feat_dim)
        self.sde = sde
        self.optimizer = optimizer(self.model.parameters())
        self.device = "cpu"

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()

    def fit(self, dataset, n_epochs, batch_size, 
            num_workers=0,
            loss_fn=None, 
            verbose=True,
            **kvargs,
            ):

        if loss_fn is None:
            loss_fn = losses.SDE_loss(
                sde=self.sde,
                model=self.model,
                optimizer=self.optimizer,
                **kvargs
            )

        return train.train(dataset, self.model, loss_fn, n_epochs, batch_size, self.device, num_workers, verbose=verbose)

    def predict(self, dataset, batch_size, num_workers=0, verbose=True):
        likelihood_fn = ood_utils.get_likelihood_fn(self.sde)
        return train.inference(dataset, self.model, likelihood_fn, batch_size, self.device, num_workers, verbose)
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    
def VESDE_Likelihood(feat_dim):
    return Likelihood(VESDE(sigma_min=0.01, sigma_max=50, N=1000), feat_dim=feat_dim)

def SubSDE_Likelihood(feat_dim):
    return Likelihood(subVPSDE(beta_min=0.1, beta_max=20, N=1000), feat_dim=feat_dim)

def VPSDE_Likelihood(feat_dim):
    return Likelihood(VPSDE(beta_min=0.1, beta_max=20, N=1000),feat_dim=feat_dim)
