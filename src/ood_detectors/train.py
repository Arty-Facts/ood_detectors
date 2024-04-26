import numpy as np
import torch
import tqdm
import ood_detectors.eval_utils as eval_utils

def train(dataset, model, loss_fn, optimizer, n_epochs, batch_size, device, num_workers=0, verbose=True, tw=None, lrs=True):
    if verbose:
        print(f'Training for {n_epochs} epochs...')
      
    data_loader = torch.eval_utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if verbose:
        epochs = tqdm.trange(n_epochs)
    else:
        epochs = range(n_epochs)
    if lrs:
        lr = optimizer.param_groups[0]['lr']
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(data_loader), epochs=n_epochs)
    scaler = torch.cuda.amp.GradScaler()
    avg_epoch_loss = []
    model.train()
    for epoch in epochs:
        avg_loss = 0.
        num_items = 0
        epoch_loss = []
        for x in data_loader:
            x = x.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = loss_fn(model, x)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if lrs:
                lr_scheduler.step()
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



def inference(dataset, model, ode_likelihood, batch_size, device, num_workers=0, verbose=True):
    data_loader = torch.eval_utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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
        bpd = ode_likelihood(x=x, score_model=model, batch_size=x.shape[0], device=device)
        all_bpds += bpd.sum()
        all_items += bpd.shape[0]
        score_id.append(bpd.cpu().numpy())

        if verbose:
            data_iter.set_description("Average bits/dim: {:5f}".format(all_bpds / all_items))

    return np.concatenate(score_id)


def eval(dataset, reference_dataset, ood_datasets, model, ode_likelihood, batch_size, device, num_workers=0, recall=0.95, verbose=True, tw=None):
    if verbose:
        print('Running eval. In-distribution data')
    score_id = inference(dataset, model, ode_likelihood, batch_size, device, num_workers, verbose=verbose)
    if verbose:
        print('Running eval. Reference data')
    score_ref = inference(reference_dataset, model, ode_likelihood, batch_size, device, num_workers, verbose=verbose)
    ref_auc, _, _ = eval_utils.auc(-score_ref, -score_id)
    ref_fpr, _ = eval_utils.fpr_recall(-score_ref, -score_id, recall)
    if verbose:
        print(f'AUC: {ref_auc:.4f}, FPR: {ref_fpr:.4f}')
    score_oods = []
    auc_oods = []
    fpr_oods = []
    for i, ood_dataset in enumerate(ood_datasets):
        if verbose:
            print(f'Running eval. Out-of-distribution data {i+1}/{len(ood_datasets)}')
        score_ood = inference(ood_dataset, model, ode_likelihood, batch_size, device, num_workers, verbose=verbose)
        score_oods.append(score_ood)
        auc_ood, _, _ = eval_utils.auc(-score_ref, -score_ood)
        auc_oods.append(auc_ood)
        fpr_ood, _ = eval_utils.fpr_recall(-score_ref, -score_ood, recall)
        fpr_oods.append(fpr_ood)
        if verbose:
            print(f'AUC: {auc_ood:.4f}, FPR: {fpr_ood:.4f}')

    return {'score': score_id, 
            'score_ref': score_ref,
            'ref_auc': ref_auc,
            'ref_fpr': ref_fpr,
            'score_oods': score_oods, 
            'auc': auc_oods, 
            'fpr': fpr_oods}