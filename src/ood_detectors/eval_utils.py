from sklearn import metrics
import numpy as np


def auc(ind_conf, ood_conf):
    """
    Calculate the Area Under the ROC Curve (AUC) given the in-distribution and out-of-distribution confidence scores.

    Args:
        ind_conf (numpy.ndarray): Confidence scores for in-distribution samples.
        ood_conf (numpy.ndarray): Confidence scores for out-of-distribution samples.

    Returns:
        float: The AUC value.
    """
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    auroc = metrics.auc(fpr, tpr)

    return auroc


def num_fp_at_recall(ind_conf, ood_conf, tpr):
    """
    Calculate the number of false positives at a given recall rate.

    Args:
        ind_conf (numpy.ndarray): Confidence scores for in-distribution samples.
        ood_conf (numpy.ndarray): Confidence scores for out-of-distribution samples.
        tpr (float): Target recall rate.

    Returns:
        int: The number of false positives.
    """
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0
    if num_ind == 0:
        return 0

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp


def fpr(ind_conf, ood_conf, tpr=0.95):
    """
    Calculate the False Positive Rate (FPR) at a given recall rate.

    Args:
        ind_conf (numpy.ndarray): Confidence scores for in-distribution samples.
        ood_conf (numpy.ndarray): Confidence scores for out-of-distribution samples.
        tpr (float, optional): Target recall rate. Defaults to 0.95.

    Returns:
        float: The False Positive Rate.
    """
    num_fp = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr


def eval_ood(
    model,
    dataset,
    reference_dataset,
    ood_datasets,
    batch_size,
    num_workers=0,
    recall=0.95,
    verbose=True,
):
    """
    Evaluate the out-of-distribution (OOD) detection performance of a model.

    Args:
        model: The model to evaluate.
        dataset: The in-distribution dataset.
        reference_dataset: The reference dataset.
        ood_datasets: List of out-of-distribution datasets.
        batch_size (int): Batch size for prediction.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        recall (float, optional): Target recall rate for FPR calculation. Defaults to 0.95.
        verbose (bool, optional): Whether to print evaluation progress. Defaults to True.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    if verbose:
        print("Running eval. In-distribution data")
    score_id = model.predict(dataset, batch_size, num_workers, verbose=verbose)
    if verbose:
        print("Running eval. Reference data")
    score_ref = model.predict(
        reference_dataset, batch_size, num_workers, verbose=verbose
    )
    ref_auc = auc(-score_ref, -score_id)
    ref_fpr = fpr(-score_ref, -score_id, recall)
    if verbose:
        print(f"AUC: {ref_auc:.4f}, FPR: {ref_fpr:.4f}")
    score_oods = []
    auc_oods = []
    fpr_oods = []
    for i, ood_dataset in enumerate(ood_datasets):
        if verbose:
            print(f"Running eval. Out-of-distribution data {i+1}/{len(ood_datasets)}")
        score_ood = model.predict(ood_dataset, batch_size, num_workers, verbose=verbose)
        score_oods.append(score_ood)
        auc_ood = auc(-score_ref, -score_ood)
        auc_oods.append(auc_ood)
        fpr_ood = fpr(-score_ref, -score_ood, recall)
        fpr_oods.append(fpr_ood)
        if verbose:
            print(f"AUC: {auc_ood:.4f}, FPR: {fpr_ood:.4f}")

    return {
        "score": score_id,
        "score_ref": score_ref,
        "ref_auc": ref_auc,
        "ref_fpr": ref_fpr,
        "score_oods": score_oods,
        "auc": auc_oods,
        "fpr": fpr_oods,
    }
