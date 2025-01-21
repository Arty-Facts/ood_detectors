import torch
import pickle
import ood_detectors.sde as sde_lib
import ood_detectors.models as models
import ood_detectors.losses as losses
import ood_detectors.likelihood as likelihood
import ood_detectors.eval_utils as eval_utils
import ood_detectors.ops_utils as ops_utils
import functools
from ood_detectors.residual import Residual
from ood_detectors.knn import KNN
import pathlib
import tqdm
import yaml
import multiprocessing as mp
import random

def run(conf, data, encoder, dataset, method, device, reduce_data_eval=-1, reduce_data_train=-1, verbose=False, checkpoints="results"):
    train_data_path = data[encoder][dataset]["id"]["train"]
    batch_size = conf.get('batch_size', 1024)
    with open(train_data_path, 'rb') as f:
        train_blob = pickle.load(f)
    data_train = train_blob['features']
    if reduce_data_train > 0:
        prem = torch.randperm(data_train.shape[0])
        data_train = data_train[prem[:reduce_data_train]]

    if method == 'Residual':
        dims = conf['dims']
        ood_model = Residual(dims=dims)
        if pathlib.Path(f"{checkpoints}/{method}_{dataset}_{encoder}.pt").exists():
            model_path = f"{checkpoints}/{method}_{dataset}_{encoder}.pt"
            checkpoint = torch.load(model_path)
            ood_model.load_state_dict(checkpoint['model'])
            loss = [checkpoint['results']['id']['loss']]
        else:
            loss = ood_model.fit(data_train)
    elif method == 'KNN':
        k = conf['k']
        ood_model = KNN(k=k)
        if pathlib.Path(f"{checkpoints}/{method}_{dataset}_{encoder}.pt").exists():
            model_path = f"{checkpoints}/{method}_{dataset}_{encoder}.pt"
            checkpoint = torch.load(model_path)
            ood_model.load_state_dict(checkpoint['model'])
            loss = [checkpoint['results']['id']['loss']]
        else:
            loss = ood_model.fit(data_train)
    else:

        # Hyperparameters
        feat_dim = data_train.shape[-1]
        n_epochs = conf.get('n_epochs', 200)
        bottleneck_channels = conf['bottleneck_channels']
        num_res_blocks = conf['num_res_blocks']
        time_embed_dim = conf['time_embed_dim']
        dropout = conf['dropout']
        lr = conf['lr']
        beta1 = conf.get('beta1', 0.9)
        beta2 = conf.get('beta2', 0.999)
        eps = conf.get('eps', 1e-8)
        weight_decay = conf.get('weight_decay', 0)
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
        if pathlib.Path(f"{checkpoints}/{method}_{dataset}_{encoder}.pt").exists():
            model_path = f"{checkpoints}/{method}_{dataset}_{encoder}.pt"
            checkpoint = torch.load(model_path)
            ood_model.load_state_dict(checkpoint['model'])
            loss = [checkpoint['results']['id']['loss']]
        else:
            loss = ood_model.fit(
                data_train,
                n_epochs=n_epochs,
                batch_size=batch_size,
                update_fn=update_fn,
                verbose=verbose,
            )

    score_id = ood_model.predict(data_train, batch_size, verbose=verbose)
    test_data_path = data[encoder][dataset]["id"]["test"]
    with open(test_data_path, 'rb') as f:
        test_blob = pickle.load(f)
    data_test = test_blob['features']
    if reduce_data_eval > 0:
        prem = torch.randperm(data_test.shape[0])
        data_test = data_test[prem[:reduce_data_eval]]

    score_ref = ood_model.predict(data_test, batch_size, verbose=verbose)
    results = {}
    scores = {
        'score_id': score_id,
        'score_ref': score_ref,
    }
    id_auc = eval_utils.auc(-score_ref, -score_id)
    id_fpr_95 = eval_utils.fpr(-score_ref, -score_id, 0.95)
    results['id'] = {'AUC': id_auc, 
                     'FPR_95': id_fpr_95, 
                     'score_id': score_id.mean().item(),
                     'score_ref': score_ref.mean().item(), 
                     'loss': loss[-1]}
    ood_auc_mean = []
    for name, datasets in data[encoder][dataset].items():
        if name == 'id':
            continue
        if name not in results:
            results[name] = {}
        for type_name, data in datasets.items():
            with open(data, 'rb') as f:
                ood_data = pickle.load(f)
            data = ood_data['features']
            if reduce_data_eval > 0:
                prem = torch.randperm(data.shape[0])
                data = data[prem[:reduce_data_eval]]
            score_ood = ood_model.predict(data, batch_size, verbose=verbose)
            ood_auc = eval_utils.auc(-score_ref, -score_ood)
            ood_fpr_95 = eval_utils.fpr(-score_ref, -score_ood, 0.95)
            results[name][type_name] = {'AUC': ood_auc, 
                                        'FPR_95': ood_fpr_95,
                                        'score_ood': score_ood.mean().item(),
                                        }
            if name not in scores:
                scores[name] = {}
            scores[name][type_name] = score_ood
            ood_auc_mean.append(ood_auc)
    mean_auc = int((sum(ood_auc_mean) / len(ood_auc_mean))*1000)
    checkpoint_path = pathlib.Path(checkpoints)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_path / f"{method}_{dataset}_{encoder}.pt"
    torch.save({
        "config": conf,
        "model": ood_model.state_dict(),
        "results": results,
        "method": method,
        "encoder": encoder,
        "dataset": dataset,
    }, model_path)

    with open(checkpoint_path/f"{method}_{dataset}_{encoder}_scores.pkl", 'wb') as f:
        pickle.dump(scores, f)
    return results

def objective(method, encoder, dataset, method_configs, checkpoints, device):
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
    if pathlib.Path(f"{checkpoints}/{method}_{encoder}_{dataset}_scores.pkl").exists():
        return
    results = {}
    conf = method_configs[method]
    res = run(conf, features_data, encoder, dataset, method, device, checkpoints=checkpoints)
    if method not in results:
        results[method] = {}
    if encoder not in results[method]:
        results[method][encoder] = {}
    if dataset not in results[method][encoder]:
        results[method][encoder][dataset] = {}
    results[method][encoder][dataset]["id"] = {
        "dataset": dataset+"_test",
        "metrics": {
            "AUC": float(res['id']['AUC']),
            "FPR_95": float(res['id']['FPR_95']),
            "score_id": float(res['id']['score_id']),
            "score_ref": float(res['id']['score_ref']),
            "loss": float(res['id']['loss']),
        },
    }
    means = {}  
    for type_name, datasets in res.items():
        if type_name == 'id':
            continue
        if type_name not in results[method][encoder]:
            results[method][encoder][dataset][type_name] = []
        for dataset_name, res in datasets.items():
            results[method][encoder][dataset][type_name].append({
                "dataset": dataset_name,
                "metrics": {
                    "AUC": float(res['AUC']),
                    "FPR_95": float(res['FPR_95']),
                    "score": float(res['score_ood']),
                },
            })
            if type_name not in means:
                means[type_name] = {}
            if "AUC" not in means[type_name]:
                means[type_name]["AUC"] = []
            if "FPR_95" not in means[type_name]:
                means[type_name]["FPR_95"] = []
            if "score" not in means[type_name]:
                means[type_name]["score"] = []
            
            means[type_name]["AUC"].append(float(res['AUC']))
            means[type_name]["FPR_95"].append(float(res['FPR_95']))
            means[type_name]["score"].append(float(res['score_ood']))

    pathlib.Path("results").mkdir(parents=True, exist_ok=True)
    with open(f"results/{method}_{encoder}_{dataset}.yaml", "w") as f:
        yaml.dump(results, f, sort_keys=False)

def main(checkpoint = "results_v6"):

    # datasets = ['imagenet', 'imagenet200', 'cifar10', 'cifar100', 'covid', 'mnist']
    datasets = ['imagenet_sub', 'imagenet', 'imagenet200', 'cifar10', 'cifar100']
    general_encoders = ['dino', 'dinov2', 'vit', 'clip']
    encoders = ['repvgg', 'resnet50d', 'swin', 'deit', 'swin_t', 'vit_b16']
    # encoders = ['dinov2', 'vit', 'clip']
    open_ood_encoders = ['resnet18_32x32_cifar10_open_ood', 'resnet18_32x32_cifar100_open_ood', 'resnet18_224x224_imagenet200_open_ood', 'resnet50_224x224_imagenet_open_ood']
    open_ood_datasets = ['cifar10', 'cifar100', 'imagenet200', 'imagenet']
    methods = ['subVPSDE', 'VESDE', 'VPSDE']
    # methods = ['Residual']
    jobs = []
    # train_config = {
    #     'n_epochs': 300,
    #     'bottleneck_channels': 768,
    #     'num_res_blocks': 10,
    #     'time_embed_dim': 512,
    #     'dropout': 0.3,
    #     'lr':5e-5,
    #     'beta1': 0.9,
    #     'beta2': 0.999,
    #     'eps': 1e-8,
    #     'weight_decay': 0,
    #     'continuous': True,
    #     'reduce_mean': True,
    #     'likelihood_weighting': False,
    # }
    train_config = {
        'n_epochs': 500,
        'bottleneck_channels': 512,
        'num_res_blocks': 5,
        'time_embed_dim': 512,
        'dropout': 0.0,
        'lr':5e-5,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'weight_decay': 0,
        'continuous': True,
        'reduce_mean': True,
        'likelihood_weighting': False,
    }


    method_configs = {
        'Residual':
            {
                'dims': None,
            },
        'VESDE':
            {
                **train_config,
                'sigma_min': 0.05,
                'sigma_max': 30,
            },
        'VPSDE':
            {
                **train_config,
                'beta_min': 0.5,
                'beta_max': 15,
            },
        'subVPSDE':
            {
                **train_config,
                'beta_min': 0.5,
                'beta_max': 15,
            },
    }

    for m in methods:
        for e in encoders+general_encoders:
            for d in datasets:
                jobs.append((m, e, d, method_configs, checkpoint))

        for e, d in zip(open_ood_encoders, open_ood_datasets):
            jobs.append((m, e, d, method_configs, checkpoint))

    
    gpu_nodes = [0, 1, 2, 3] * 2
    random.shuffle(jobs)
    print(f'Running {len(jobs)} jobs...')
    ops_utils.parallelize(objective, jobs, gpu_nodes, verbose=True, timeout=60*60*24)


if __name__ == '__main__':
    mp.freeze_support()
    main()
