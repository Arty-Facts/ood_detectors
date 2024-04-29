from sklearn import metrics
import numpy as np

def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    # aupr_in = metrics.auc(recall_in, precision_in)
    # aupr_out = metrics.auc(recall_out, precision_out)

    return auroc

def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp

def fpr(ind_conf, ood_conf, tpr=0.95):
    num_fp = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr

def eval_ood(model, dataset, reference_dataset, ood_datasets, batch_size, num_workers=0, recall=0.95, verbose=True):
    if verbose:
        print('Running eval. In-distribution data')
    score_id = model.predict(dataset, batch_size, num_workers, verbose=verbose)
    if verbose:
        print('Running eval. Reference data')
    score_ref = model.predict(reference_dataset, batch_size, num_workers, verbose=verbose)
    ref_auc = auc(-score_ref, -score_id)
    ref_fpr = fpr(-score_ref, -score_id, recall)
    if verbose:
        print(f'AUC: {ref_auc:.4f}, FPR: {ref_fpr:.4f}')
    score_oods = []
    auc_oods = []
    fpr_oods = []
    for i, ood_dataset in enumerate(ood_datasets):
        if verbose:
            print(f'Running eval. Out-of-distribution data {i+1}/{len(ood_datasets)}')
        score_ood = model.predict(ood_dataset, batch_size, num_workers, verbose=verbose)
        score_oods.append(score_ood)
        auc_ood = auc(-score_ref, -score_ood)
        auc_oods.append(auc_ood)
        fpr_ood = fpr(-score_ref, -score_ood, recall)
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