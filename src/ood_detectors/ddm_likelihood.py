from ood_detectors.models.mlp import SimpleMLP
import ood_detectors.ood_utils as ood_utils
import ood_detectors.losses as losses
import ood_detectors.train as train
import torch
import functools

class DDM_Likelihood():
    def __init__(self, 
                    feat_dim,
                    bottleneck_channels=512,
                    dropout=0.25,
                    num_res_blocks=10,
                    sigma = 25,
                    time_embed_dim=512,
                 ):
        self.feat_dim = feat_dim
        self.sigma = sigma
        self.device = 'cpu'
        marginal_prob_std_fn = functools.partial(ood_utils.marginal_prob_std, sigma=self.sigma, device=self.device)
        self.model = SimpleMLP(marginal_prob_std=marginal_prob_std_fn,
                               in_channels=feat_dim,
                               time_embed_dim=time_embed_dim,
                               model_channels=feat_dim,
                               bottleneck_channels=bottleneck_channels,
                               out_channels=feat_dim,
                               num_res_blocks=num_res_blocks,
                               dropout=dropout)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()

    def fit(self, dataset, n_epochs, batch_size, num_workers=0):
        self.model.train()
        marginal_prob_std_fn = functools.partial(ood_utils.marginal_prob_std, sigma=self.sigma, device=self.device)
        loss = functools.partial(losses.score_based_loss, marginal_prob_std=marginal_prob_std_fn)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        result = train.train(dataset, self.model, loss, optimizer, n_epochs, batch_size, self.device, num_workers)
        return result
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, dataset, batch_size, num_workers=0, verbose=True):
        self.model.eval()
        marginal_prob_std_fn = functools.partial(ood_utils.marginal_prob_std, sigma=self.sigma, device=self.device)
        diffusion_coeff_fn = functools.partial(ood_utils.diffusion_coeff, sigma=self.sigma, device=self.device)
        ode_likelihood = functools.partial(ood_utils.ode_likelihood, 
                                        marginal_prob_std=marginal_prob_std_fn,
                                        diffusion_coeff=diffusion_coeff_fn)
        
        return train.inference(dataset, self.model, ode_likelihood, batch_size, self.device, num_workers=num_workers, verbose=verbose)


