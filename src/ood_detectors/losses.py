import torch
import numpy as np
import ood_detectors.sde as sde_lib
import ood_detectors.ood_utils as ood_utils
import ood_detectors.ema as ema



def get_sde_loss_fn(
    sde, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5
):
    """Create a loss function for training with arbirary SDEs.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      train: `True` for training loss and `False` for evaluation loss.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses
        according to https://arxiv.org/abs/2101.09258.

    Returns:
      A loss function.
    """
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = ood_utils.get_score_fn(sde, model, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None] * z
        score = score_fn(perturbed_data, t)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_smld_loss_fn(vesde, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, sde_lib.VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None]
        perturbed_data = noise + batch
        score = model(perturbed_data, labels)
        target = -noise / (sigmas**2)[:, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas**2
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_ddpm_loss_fn(vpsde, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, sde_lib.VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = (
        torch.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    )

    def loss_fn(model, batch):
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = (
            sqrt_alphas_cumprod[labels, None] * batch
            + sqrt_1m_alphas_cumprod[labels, None] * noise
        )
        score = model(perturbed_data, labels)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


class SDE_EMA_Warmup_GradClip:
    def __init__(
        self,
        sde,
        model,
        optimizer,
        ema_rate=0.999,
        warmup=5000,
        grad_clip=1,
        reduce_mean=True,
        continuous=True,
        likelihood_weighting=False,
    ):
        """Create a one-step training/evaluation class.
        """
        if continuous:
            self.loss_fn = get_sde_loss_fn(
                sde,
                reduce_mean=reduce_mean,
                continuous=True,
                likelihood_weighting=likelihood_weighting,
            )
        else:
            assert (
                not likelihood_weighting
            ), "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(sde, sde_lib.VESDE):
                self.loss_fn = get_smld_loss_fn(sde, reduce_mean=reduce_mean)
            elif isinstance(sde, sde_lib.VPSDE):
                self.loss_fn = get_ddpm_loss_fn(sde, reduce_mean=reduce_mean)
            else:
                raise ValueError(
                    f"Discrete training for {sde.__class__.__name__} is not recommended."
                )

        self.optimizer = optimizer
        self.ema = ema.ExponentialMovingAverage(model.parameters(), decay=ema_rate)
        self.step = 0
        self.warmup = warmup
        self.grad_clip = grad_clip
        self.lr = optimizer.param_groups[0]["lr"]
        # self.scaler = torch.amp.GradScaler()

    def __call__(self, model, x, train=True):
        """Running one step of training or evaluation.
        Returns:
        loss: The average loss value of this state.
        """

        if train:
            self.optimizer.zero_grad()
            loss = self.loss_fn(model, x)
            loss.backward()
            if self.warmup > 0:
                for g in self.optimizer.param_groups:
                    g["lr"] = self.lr * np.minimum(self.step / self.warmup, 1.0)
            if self.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.grad_clip
                )
            self.optimizer.step()
            self.step += 1
            if self.ema is not None:
                self.ema.update(model.parameters())
        else:
            with torch.no_grad():
                if self.ema is not None:
                    self.ema.store(model.parameters())
                    self.ema.copy_to(model.parameters())
                loss = self.loss_fn(model, x)
                if self.ema is not None:
                    self.ema.restore(model.parameters())
        return loss


class SDE_LRS_BF16:
    def __init__(
        self,
        sde,
        model,
        optimizer,
        total_steps=100000,
        reduce_mean=True,
        continuous=True,
        likelihood_weighting=False,
    ):
        """Create a one-step training/evaluation class.
        """
        if continuous:
            self.loss_fn = get_sde_loss_fn(
                sde,
                reduce_mean=reduce_mean,
                continuous=True,
                likelihood_weighting=likelihood_weighting,
            )
        else:
            assert (
                not likelihood_weighting
            ), "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(sde, sde_lib.VESDE):
                self.loss_fn = get_smld_loss_fn(sde, reduce_mean=reduce_mean)
            elif isinstance(sde, sde_lib.VPSDE):
                self.loss_fn = get_ddpm_loss_fn(sde, reduce_mean=reduce_mean)
            else:
                raise ValueError(
                    f"Discrete training for {sde.__class__.__name__} is not recommended."
                )

        self.optimizer = optimizer
        self.lr = optimizer.param_groups[0]["lr"]
        self.scaler = torch.amp.GradScaler()
        self.total_steps = total_steps
        self.lrs = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=self.total_steps,
        )

    def __call__(self, model, x, *args, **kwargs):
        """Running one step of training or evaluation.
        Returns:
        loss: The average loss value of this state.
        """

        self.optimizer.zero_grad()
        with torch.amp.autocast(dtype=torch.bfloat16):
            loss = self.loss_fn(model, x)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lrs.step()


        return loss
    
class SDE_BF16:
    def __init__(
        self,
        sde,
        model,
        optimizer,
        reduce_mean=True,
        continuous=True,
        likelihood_weighting=False,
    ):
        """Create a one-step training/evaluation class.
        """
        if continuous:
            self.loss_fn = get_sde_loss_fn(
                sde,
                reduce_mean=reduce_mean,
                continuous=True,
                likelihood_weighting=likelihood_weighting,
            )
        else:
            assert (
                not likelihood_weighting
            ), "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(sde, sde_lib.VESDE):
                self.loss_fn = get_smld_loss_fn(sde, reduce_mean=reduce_mean)
            elif isinstance(sde, sde_lib.VPSDE):
                self.loss_fn = get_ddpm_loss_fn(sde, reduce_mean=reduce_mean)
            else:
                raise ValueError(
                    f"Discrete training for {sde.__class__.__name__} is not recommended."
                )

        self.optimizer = optimizer
        self.scaler = torch.amp.GradScaler()
      
    def __call__(self, model, x, *args, **kwargs):
        """Running one step of training or evaluation.
        Returns:
        loss: The average loss value of this state.
        """

        self.optimizer.zero_grad()
        with torch.amp.autocast(dtype=torch.bfloat16):
            loss = self.loss_fn(model, x)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss
class SDE_F16:
    def __init__(
        self,
        sde,
        model,
        optimizer,
        reduce_mean=True,
        continuous=True,
        likelihood_weighting=False,
    ):
        """Create a one-step training/evaluation class.
        """
        if continuous:
            self.loss_fn = get_sde_loss_fn(
                sde,
                reduce_mean=reduce_mean,
                continuous=True,
                likelihood_weighting=likelihood_weighting,
            )
        else:
            assert (
                not likelihood_weighting
            ), "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(sde, sde_lib.VESDE):
                self.loss_fn = get_smld_loss_fn(sde, reduce_mean=reduce_mean)
            elif isinstance(sde, sde_lib.VPSDE):
                self.loss_fn = get_ddpm_loss_fn(sde, reduce_mean=reduce_mean)
            else:
                raise ValueError(
                    f"Discrete training for {sde.__class__.__name__} is not recommended."
                )

        self.optimizer = optimizer
        self.scaler = torch.amp.GradScaler()
      
    def __call__(self, model, x, *args, **kwargs):
        """Running one step of training or evaluation.
        Returns:
        loss: The average loss value of this state.
        """

        self.optimizer.zero_grad()
        with torch.amp.autocast(dtype=torch.float16):
            loss = self.loss_fn(model, x)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss
class SDE_F32:
    def __init__(
        self,
        sde,
        model,
        optimizer,
        reduce_mean=True,
        continuous=True,
        likelihood_weighting=False,
    ):
        """Create a one-step training/evaluation class.
        """
        if continuous:
            self.loss_fn = get_sde_loss_fn(
                sde,
                reduce_mean=reduce_mean,
                continuous=True,
                likelihood_weighting=likelihood_weighting,
            )
        else:
            assert (
                not likelihood_weighting
            ), "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(sde, sde_lib.VESDE):
                self.loss_fn = get_smld_loss_fn(sde, reduce_mean=reduce_mean)
            elif isinstance(sde, sde_lib.VPSDE):
                self.loss_fn = get_ddpm_loss_fn(sde, reduce_mean=reduce_mean)
            else:
                raise ValueError(
                    f"Discrete training for {sde.__class__.__name__} is not recommended."
                )

        self.optimizer = optimizer
      
    def __call__(self, model, x, *args, **kwargs):
        """Running one step of training or evaluation.
        Returns:
        loss: The average loss value of this state.
        """

        self.optimizer.zero_grad()
        loss = self.loss_fn(model, x)
        loss.backward()
        self.optimizer.step()
        return loss