import optuna
import pathlib
import ood_detectors.ops_utils as ops_utils
from evaluate import run
import multiprocessing as mp
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
        conf['n_epochs'] = trial.suggest_int('n_epochs', 100, 500, step=100)
        conf['bottleneck_channels'] = trial.suggest_int('bottleneck_channels', 256, 2048, step = 256)
        conf['num_res_blocks'] = trial.suggest_int('num_res_blocks', 3, 15)
        conf['time_embed_dim'] = trial.suggest_int('time_embed_dim', 256, 1024, step = 256)
        conf['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        conf['lr'] = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
        # conf['beta1'] = trial.suggest_float('beta1', 0.5, 0.999)
        # conf['beta2'] = trial.suggest_float('beta2', 0.9, 0.999)
        # conf['eps'] = trial.suggest_float('eps', 1e-12, 1e-6, log=True)
        # conf['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 1e-3)
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
            conf['sigma_min'] = trial.suggest_float('sigma_min', 0.01, 0.1)
            conf['sigma_max'] = trial.suggest_int('beta_max', 10, 60, step=5)
        elif method == 'VPSDE':
            conf['beta_min'] = trial.suggest_float('beta_min', 0.0, 1.0)
            conf['beta_max'] = trial.suggest_int('beta_max', 10, 30, step=5)
        elif method == 'subVPSDE':
            conf['beta_min'] = trial.suggest_float('beta_min', 0.0, 1.0)
            conf['beta_max'] = trial.suggest_int('beta_max', 10, 30, step=5)
        else:
            raise ValueError(f'Unknown method: {method}')
    return conf


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
                bar.set_description(f'Method: {method}, Encoder: {encoder}, Dataset: {dataset}')
            results = run(conf, data, encoder, dataset, method, device)
            auc = results['id']["AUC"]
            fpr = results['id']["FPR_95"]
            loss = results['id']['loss']
            score_id = results['id']['score_id']
            score_ref = results['id']['score_ref']
            farood = sum([v["AUC"] for v in results['farood'].values()]) / len(results['farood'])
            nearood = sum([v["AUC"] for v in results['nearood'].values()]) / len(results['nearood'])
            ids.append(auc)
            faroods.append(farood)
            nearoods.append(nearood)
            trial.set_user_attr(f'{encoder}_{dataset}_id_AUC', float(auc))
            trial.set_user_attr(f'{encoder}_{dataset}_id_FPR95', float(fpr))
            trial.set_user_attr(f'{encoder}_{dataset}_id_loss', float(loss))
            trial.set_user_attr(f'{encoder}_{dataset}_score_id', float(score_id))
            trial.set_user_attr(f'{encoder}_{dataset}_score_ref', float(score_ref))
            for d_name, v in results['farood'].items():
                for m, value in v.items():
                    trial.set_user_attr(f'{encoder}_{dataset}_farood_{d_name}_{m}', float(value))
            for d_name, v in results['nearood'].items():
                for m, value in v.items():
                    trial.set_user_attr(f'{encoder}_{dataset}_nearood_{d_name}_{m}', float(value))
                
            if verbose:
                bar.set_postfix(id=auc, farood=farood, nearood=nearood)
                bar.update()
    id = sum(ids) / len(ids)
    farood = sum(faroods) / len(faroods)
    nearood = sum(nearoods) / len(nearoods)
    return abs(id-0.5), farood, nearood

def ask_tell_optuna(objective_func, data, encoders, datasets, method, device):
    study_name = f'{method}'
    db = f'sqlite:///optuna_v2.db'
    study = optuna.create_study(directions=['minimize', 'maximize', 'maximize'], study_name=study_name, storage=db, load_if_exists=True)
    trial = study.ask()
    res = objective_func(trial, data, encoders, datasets, method, device)
    study.tell(trial, res)
        

def main():
    # features = pathlib.Path(r"H:\arty\data\features_opt")
    features = pathlib.Path("/mnt/data/arty/data/features_opt")
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
            tmp[parts[-2]] = path
    # encoders = ['repvgg', 'resnet50d', 'swin', 'deit', 'dino', 'dinov2', 'vit', 'clip']
    encoders = ['repvgg', 'resnet50d', 'swin', 'deit', 'dino', 'dinov2', 'vit']
    #['resnet18_32x32_cifar10_open_ood', 'resnet18_32x32_cifar100_open_ood', 'resnet18_224x224_imagenet200_open_ood', 'resnet50_224x224_imagenet_open_ood']
    # datasets = ['imagenet', 'imagenet200', 'cifar10', 'cifar100', 'covid', 'mnist']
    datasets = ['imagenet']
    # methods = ['VESDE', 'VPSDE', 'subVPSDE', 'Residual']
    methods = ['VESDE', 'VPSDE',]
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
