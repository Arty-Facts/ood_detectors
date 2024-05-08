import numpy as np
import torch
import tqdm

def train(dataset, model, update_fn, n_epochs, batch_size, device, num_workers=0, verbose=True, tw=None, collate_fn=None):
    """
    Trains a model on a given dataset for a specified number of epochs.

    Args:
        dataset (torch.utils.data.Dataset): The training dataset.
        model (torch.nn.Module): The model to be trained.
        update_fn (callable): The function that updates the model parameters.
        n_epochs (int): The number of training epochs.
        batch_size (int): The batch size for training.
        device (torch.device): The device to run the training on.
        num_workers (int, optional): The number of workers for data loading. Defaults to 0.
        verbose (bool, optional): Whether to display training progress. Defaults to True.
        tw (tensorboardX.SummaryWriter, optional): TensorboardX SummaryWriter for logging. Defaults to None.
        lrs (bool, optional): Whether to use learning rate scheduling. Defaults to True.

    Returns:
        list: A list of average epoch losses.
    """
    if verbose:
        print(f'Training for {n_epochs} epochs...')
    if collate_fn is None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
    if verbose:
        epochs = tqdm.trange(n_epochs)
    else:
        epochs = range(n_epochs)
    avg_epoch_loss = []
    model.train()
    for epoch in epochs:
        avg_loss = 0.
        num_items = 0
        epoch_loss = []
        for x in data_loader:
            x = x.to(device)
            loss = update_fn(model, x)
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            epoch_loss.append(loss.item())
        epoch_loss = np.mean(epoch_loss)
        avg_epoch_loss.append(epoch_loss)
        if tw is not None:
            tw.add_scalar('Loss/train', epoch_loss, epoch)
        # Print the averaged training loss so far.
        if verbose:
            epochs.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    return avg_epoch_loss


def inference(dataset, model, ode_likelihood, batch_size, device, num_workers=0, verbose=True, collate_fn=None):
    """
    Performs inference using a trained model on a given dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset for inference.
        model (torch.nn.Module): The trained model.
        ode_likelihood (callable): The function that computes the likelihood.
        batch_size (int): The batch size for inference.
        device (torch.device): The device to run the inference on.
        num_workers (int, optional): The number of workers for data loading. Defaults to 0.
        verbose (bool, optional): Whether to display inference progress. Defaults to True.

    Returns:
        numpy.ndarray: An array of inference scores.
    """
    if collate_fn is None:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    all_bpds = 0.
    all_items = 0
    score_id = []
    if verbose:
        data_iter = tqdm.tqdm(data_loader)
    else:
        data_iter = data_loader
    model.eval()

    for x in data_iter:
        x = x.to(device)
        bpd = ode_likelihood(model=model, x=x)
        all_bpds += bpd.sum()
        all_items += bpd.shape[0]
        score_id.append(bpd.cpu().numpy())

        if verbose:
            data_iter.set_description("Average bits/dim: {:5f}".format(all_bpds / all_items))

    return np.concatenate(score_id)

