import torch
import numpy as np
from scipy import integrate
from sklearn import metrics
import pickle
import time
import multiprocessing as mp
import random
from torchdiffeq import odeint as odeint_torch

def marginal_prob_std(time_step, sigma, device="cuda"):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:
    time_step: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The standard deviation.
  """
  if not isinstance(time_step, (torch.Tensor, torch.cuda.FloatTensor)):
    time_step = torch.tensor(time_step, device=device)
  return torch.sqrt((sigma**(2 * time_step) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(time_step, sigma, device="cuda"):
  """Compute the diffusion coefficient of our SDE.

  Args:
    time_step: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The vector of diffusion coefficients.
  """
  if not isinstance(time_step, (torch.Tensor, torch.cuda.FloatTensor)):
    time_step = torch.tensor(time_step, device=device)
  return sigma**time_step

def prior_likelihood(z, sigma):
  """The likelihood of a Gaussian distribution with mean zero and
      standard deviation sigma."""
  shape = z.shape
  N = np.prod(shape[1:])
  return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=1) / (2 * sigma**2)


def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64, 
                device='cuda',
                z=None,
                end=1e-3,
                start=1):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, 768, device=device) \
      * marginal_prob_std(t)[:, None]
  else:
    init_x = z
    
  shape = init_x.shape

#   def score_eval_wrapper(sample, time_steps):
#     """A wrapper for evaluating the score-based model for the black-box ODE solver."""
#     sample = sample.reshape(shape)
#     time_steps = time_steps.reshape((sample.shape[0], ))
#     with torch.no_grad():
#         score = score_model(sample, time_steps)
#     return score

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
    # time_steps = torch.ones((shape[0],), device=device) * t
    # g = diffusion_coeff(t)
    # return -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

  
#Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (start, end), init_x.reshape(-1).cpu().numpy(), rtol=1e-5, atol=1e-5, method='RK45')  
  #print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
#   timesteps = torch.tensor([start, end], device=device) 
#   res = odeint_torch(ode_func, init_x, timesteps, rtol=1e-5, atol=1e-5, method='fehlberg2')
#   x = res[-1].reshape(shape)
  return x


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