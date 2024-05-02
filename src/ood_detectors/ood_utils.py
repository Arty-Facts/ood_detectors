import torch
import numpy as np
from scipy import integrate
from torchdiffeq import odeint as odeint_torch
import ood_detectors.sde as sde_lib



def ode_likelihood(x,
                   score_model,
                   marginal_prob_std,
                   diffusion_coeff,
                   batch_size=64, #TODO: we are not using this
                   device='cuda',
                   eps=1e-5):
    """Compute the likelihood with probability flow ODE.

    Args:
        x: Input data.
        score_model: A PyTorch model representing the score-based model.
        marginal_prob_std: A function that gives the standard deviation of the
        perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the
        forward SDE.
        batch_size: The batch size. Equals to the leading dimension of `x`.
        device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
        eps: A `float` number. The smallest time step for numerical stability.

    Returns:
        z: The latent code for `x`.
        bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    epsilon = torch.randn_like(x)

    def divergence_eval(sample, time_steps, epsilon):
        """Compute the divergence of the score-based model with Skilling-Hutchinson."""
        with torch.enable_grad():
            sample.requires_grad_(True)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample)[0]
        return torch.sum(grad_score_e * epsilon, dim=1)

    shape = x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the black-box ODE solver."""
        sample = sample.reshape(shape)
        time_steps = time_steps.reshape((sample.shape[0], ))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score
    
    def divergence_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
        with torch.no_grad():
            # Obtain x(t) by solving the probability flow ODE.
            sample = sample.reshape(shape)
            time_steps = time_steps.reshape((sample.shape[0], ))
            # Compute likelihood.
            div = divergence_eval(sample, time_steps, epsilon)
        return div

    def ode_func(t, x):
        """The ODE function for the black-box solver."""
        time_steps = torch.ones((shape[0],), device=device) * t
        sample = x[:-shape[0]]
        logp = x[-shape[0]:]
        g = diffusion_coeff(t)
        sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
        return torch.cat([sample_grad.reshape(-1), logp_grad.reshape(-1)], dim=0)

    init_state = torch.cat([x.reshape(-1), torch.zeros(x.size(0), device=device)], dim=0)  # Concatenate x (flattened) and logp
    timesteps = torch.tensor([eps, 1.0], device=device)
    # Solve the ODE
    # 'dopri8' 7s
    # 'dopri5' 1.9s - good same as scipy.solve_ivp rk45
    # 'bosh3' 2.5s
    # 'fehlberg2' 1.4s - is scipy.solve_ivp rkf45
    # 'adaptive_heun' 4s
    # 'euler' nan
    # 'midpoint' nan
    # 'rk4' 1s inaccurate 
    # 'explicit_adams' 1s inaccurate 
    # 'implicit_adams' 1s inaccurate
    # 'fixed_adams' 1s inaccurate
    # 'scipy_solver'
    res = odeint_torch(ode_func, init_state, timesteps, rtol=1e-5, atol=1e-5, method='fehlberg2')
    zp = res[-1]

    z = zp[:-shape[0]].reshape(shape)
    delta_logp = zp[-shape[0]:].reshape(shape[0])
    sigma_max = marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max) #TODO: do we need this?
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[1:])
    bpd = bpd / N + 8.
    return bpd


def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """

  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None]
      return score

  elif isinstance(sde, sde_lib.VESDE):
    def score_fn(x, t):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model(x, labels)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


def get_likelihood_fn(
    sde, hutchinson_type="Rademacher", rtol=1e-5, atol=1e-5, method="fehlberg2", eps=1e-5
):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      inverse_scaler: The inverse data normalizer.
      hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
      rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
      atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
      method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
      eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
      A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    def drift_fn(model, x, t):
        """The drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def div_fn(model, x, t, noise):
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

    def likelihood_fn(model, x):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
          model: A score model.
          x: A PyTorch tensor.

        Returns:
          bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
          z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
          nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        device = x.device
        with torch.no_grad():
            shape = x.shape
            if hutchinson_type == "Gaussian":
                epsilon = torch.randn_like(x)
            elif hutchinson_type == "Rademacher":
                epsilon = torch.randint_like(x, low=0, high=2).float() * 2 - 1.0
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, data):
                sample = data[:shape.numel()].clone().reshape(shape).float()

                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = drift_fn(model, sample, vec_t)
                div = div_fn(model, sample, vec_t, epsilon)
                return torch.cat([drift.reshape(-1), div], 0)
            
            init_state = torch.cat([x.reshape(-1), torch.zeros(shape[0], device=device)], 0)
            timesteps = torch.tensor([eps, sde.T], device=device)

            # Solving the ODE
            res = odeint_torch(ode_func, init_state, timesteps, rtol=rtol, atol=atol, method=method)
            zp = res[-1]

            z = zp[:-shape[0]].reshape(shape)
            prior_logp = sde.prior_logp(z)
           
            delta_logp = zp[-shape[0]:].reshape(shape[0])

            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = torch.prod(torch.tensor(shape[1:], device=device)).item()
            bpd = bpd / N + 8  # Convert log-likelihood to bits/dim
            return bpd

    return likelihood_fn