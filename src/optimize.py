import ood_detectors.sde as sde_lib 
import ood_detectors.models as models
import ood_detectors.losses as losses
import ood_detectors.vision as vision
import ood_detectors.likelihood as likelihood
from ood_detectors.residual import Residual
import ood_detectors.eval_utils as eval_utils
import optuna
import pathlib
import functools
import torch
import ood_detectors.ops_utils as ops_utils
import multiprocessing as mp
import pickle
import random
import tqdm
# data = {
#         'features': features_vectors,
#         'encoder': encoder_name,
#         'target_dataset': dataset_name,
#         'dataset': name,
#         'type': type_name
#     }

def select_trial(trial,method):
    conf = {}
    if method == 'Residual':
        conf['dims'] = trial.suggest_float('Residual.dims', 0, 1)
    else:
        conf['n_epochs'] = trial.suggest_int('n_epochs', 100, 1000, step=100)
        conf['bottleneck_channels'] = trial.suggest_int('bottleneck_channels', 256, 2048, step = 256)
        conf['num_res_blocks'] = trial.suggest_int('num_res_blocks', 3, 15)
        conf['time_embed_dim'] = trial.suggest_int('time_embed_dim', 256, 1024, step = 256)
        conf['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        conf['lr'] = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
        conf['beta1'] = trial.suggest_float('beta1', 0.5, 0.999)
        conf['beta2'] = trial.suggest_float('beta2', 0.9, 0.999)
        conf['eps'] = trial.suggest_float('eps', 1e-12, 1e-6, log=True)
        conf['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 1e-3)
        if method != 'subVPSDE':
            conf['continuous'] = trial.suggest_categorical('continuous', [True, False])
        else:
            conf['continuous'] = True
        if conf['continuous']:
            conf['likelihood_weighting'] = trial.suggest_categorical('likelihood_weighting', [True, False])
        else:
            conf['likelihood_weighting'] = False
        conf['reduce_mean'] = trial.suggest_categorical('reduce_mean', [True, False])
        if method == 'VESDE':
            conf['sigma_min'] = trial.suggest_float('VESDE.sigma_min', 0.01, 0.1)
            conf['sigma_max'] = trial.suggest_float('VESDE.sigma_max', 10.0, 60.0)
        elif method == 'VPSDE':
            conf['beta_min'] = trial.suggest_float('VPSDE.beta_min', 0.0, 1.0)
            conf['beta_max'] = trial.suggest_float('VPSDE.beta_max', 10.0, 30.0)
        elif method == 'subVPSDE':
            conf['beta_min'] = trial.suggest_float('subVPSDE.beta_min', 0.0, 1.0)
            conf['beta_max'] = trial.suggest_float('subVPSDE.beta_max', 10.0, 30.0)
        else:
            raise ValueError(f'Unknown method: {method}')
    return conf

def run(conf, data, encoder, dataset, method, device, reduce_data=-1):
    train_data_path = data[encoder][dataset]["id"]["train"]
    batch_size = conf.get('batch_size', 1024)
    with open(train_data_path, 'rb') as f:
        train_blob = pickle.load(f)

    if method == 'Residual':
        dims = conf['dims']
        ood_model = Residual(dims=dims)
        ood_model.fit(train_blob['features'])
    else:

        # Hyperparameters
        feat_dim = train_blob['features'].shape[-1]
        n_epochs = conf['n_epochs']
        bottleneck_channels = conf['bottleneck_channels']
        num_res_blocks = conf['num_res_blocks']
        time_embed_dim = conf['time_embed_dim']
        dropout = conf['dropout']
        lr = conf['lr']
        beta1 = conf['beta1']
        beta2 = conf['beta2']
        eps = conf['eps']
        weight_decay = conf['weight_decay']
        continuous = conf['continuous']
        reduce_mean = conf['reduce_mean']
        likelihood_weighting = conf['likelihood_weighting']

        if method == 'VESDE':
            sigma_min = conf['sigma_min']
            sigma_max = conf['sigma_max']
            sde = sde_lib.VESDE(sigma_min=sigma_min, sigma_max=sigma_max)
        elif method == 'VPSDE':
            beta_min = conf['beta_min']
            beta_max = conf['beta_max']
            sde = sde_lib.VPSDE(beta_min=beta_min, beta_max=beta_max)
        elif method == 'subVPSDE':
            beta_min = conf['beta_min']
            beta_max = conf['beta_max']
            sde = sde_lib.subVPSDE(beta_min=beta_min, beta_max=beta_max)

        model = models.SimpleMLP(
            channels=feat_dim,
            bottleneck_channels=bottleneck_channels,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
        )

        optimizer = functools.partial(
                        torch.optim.Adam,
                        lr=lr,
                        betas=(beta1, beta2),
                        eps=eps,
                        weight_decay=weight_decay,
                        )

        ood_model = likelihood.Likelihood(
            sde = sde,
            model = model,
            optimizer = optimizer,
            ).to(device)

        update_fn = functools.partial(
            losses.SDE_BF16, 
            continuous=continuous,
            reduce_mean=reduce_mean,
            likelihood_weighting=likelihood_weighting,
            )
        
        loss = ood_model.fit(
            train_blob['features'],
            n_epochs=n_epochs,
            batch_size=batch_size,
            update_fn=update_fn,
            verbose=False,
        )

    score_id = ood_model.predict(train_blob['features'], batch_size, verbose=False)
    test_data_path = data[encoder][dataset]["id"]["test"]
    with open(test_data_path, 'rb') as f:
        test_blob = pickle.load(f)
    score_ref = ood_model.predict(test_blob['features'], batch_size, verbose=False)
    results = {}
    id_auc = eval_utils.auc(-score_ref, -score_id)
    results['id'] = id_auc
    for name, datasets in data[encoder][dataset].items():
        if name == 'id':
            continue
        if name not in results:
            results[name] = {}
        for type_name, data in datasets.items():
            with open(data, 'rb') as f:
                ood_data = pickle.load(f)
                data = ood_data['features']
            if reduce_data > 0:
                prem = torch.randperm(data.shape[0])
                data = data[prem[:reduce_data]]
            score_ood = ood_model.predict(data, batch_size, verbose=False)
            ood_auc = eval_utils.auc(-score_ref, -score_ood)
            results[name][type_name] = ood_auc
    return results

def objective(trial, data, encoders, datasets, method, device, verbose=True):
    conf = select_trial(trial, method)
    ids = []
    faroods = []
    nearoods = []
    if verbose:
        bar = tqdm.tqdm(total=len(encoders)*len(datasets))
    random.shuffle(encoders)
    for encoder in encoders:
        random.shuffle(datasets)
        for dataset in datasets:
            if verbose:
                bar.set_description(f'Method: {method},Encoder: {encoder}, Dataset: {dataset}')
            results = run(conf, data, encoder, dataset, method, device)
            id = results['id']
            farood = sum(results['farood'].values()) / len(results['farood'])
            nearood = sum(results['nearood'].values()) / len(results['nearood'])
            ids.append(id)
            faroods.append(farood)
            nearoods.append(nearood)
            if verbose:
                bar.set_postfix(id=id, farood=farood, nearood=nearood)
                bar.update()
    id = sum(ids) / len(ids)
    farood = sum(faroods) / len(faroods)
    nearood = sum(nearoods) / len(nearoods)
    return abs(id-0.5), farood, nearood

def ask_tell_optuna(objective_func, data, encoders, datasets, method, device):
    study_name = f'{method}'
    db = f'sqlite:///optuna.db'
    study = optuna.create_study(directions=['minimize', 'maximize', 'maximize'], study_name=study_name, storage=db, load_if_exists=True)
    trial = study.ask()
    res = objective_func(trial, data, encoders, datasets, method, device)
    study.tell(trial, res)
        

def main():
    # features = pathlib.Path(r"H:\arty\data\features_open_ood")
    features = pathlib.Path("/mnt/data/arty/data/features_open_ood")
    features_data = {}
    all_pkl = list(features.rglob("*.pkl"))
    for path in all_pkl:
        parts = path.parts[len(features.parts):]
        tmp = features_data
        for p in parts[:-2]:
            if p not in tmp:
                tmp[p] = {}
            tmp = tmp[p]
        else:
            with open(path, "rb") as f:
                tmp[parts[-2]] = path
    encoders = ['repvgg', 'resnet50d', 'swin', 'deit', 'dino', 'dinov2', 'vit', 'clip']
    #['resnet18_32x32_cifar10_open_ood', 'resnet18_32x32_cifar100_open_ood', 'resnet18_224x224_imagenet200_open_ood', 'resnet50_224x224_imagenet_open_ood']
    # datasets = ['imagenet', 'imagenet200', 'cifar10', 'cifar100', 'covid', 'mnist']
    datasets = ['imagenet200', 'cifar10', 'cifar100']
    methods = ['VESDE', 'VPSDE', 'subVPSDE', 'Residual']
    jobs = []
    for m in methods:
        jobs.append((objective, features_data, encoders, datasets, m))
       
    trials = 100
    gpu_nodes = [0, 1, 2, 3] * 2
    random.shuffle(jobs)
    print(f'Running {len(jobs)} jobs...')
    ops_utils.parallelize(ask_tell_optuna, jobs*trials, gpu_nodes, verbose=True, timeout=60*60*24)

if __name__ == '__main__':
    mp.freeze_support()
    main()
