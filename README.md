# ood_detectors

OOD Detectors is a Python package that offers a suite of algorithms designed to identify out-of-distribution samples in datasets. This is crucial for maintaining the reliability and accuracy of machine learning models when faced with unfamiliar data.


[![PyPI - Version](https://img.shields.io/pypi/v/ood_detectors.svg)](https://pypi.org/project/ood_detectors)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ood_detectors.svg)](https://pypi.org/project/ood_detectors)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

To install OOD Detectors, run the following command:

```console
pip install ood_detectors
```

## Usage
This package includes several OOD detection algorithms, each tailored to different aspects of OOD detection:

- Likelihood Based: SubSDE_DDM, VPSDE_DDM and VESDE_DDM are likelihood-based methods that use different variations stochastic differential equations for DDMS to detect OOD samples. 

- Residual: This method employs the least significant eigen vector for OOD detection.

All detectors share a common interface:

1. Initialize the detector with necessary hyperparameters.
2. Fit the model using fit() with the training data.
3. Use predict() to obtain OOD scores for new data samples.

## Example
```python
import ood_detectors.likelihood as likelihood

ood_detector = likelihood.SubSDE_DDM(feat_dim).to('cuda')
train_loss = ood_detector.fit(train_data, n_epochs, batch_size)
scores = ood_detector.predict(test_data, batch_size)
```

```python
from ood_detectors import Residual

ood_detector = Residual()
train_loss = ood_detector.fit(train_data)
scores = ood_detector.predict(test_data)
```

## low-level interface

The low-level interface allows you to customize the training process and access the model's internal components.

```python
import ood_detectors.likelihood as likelihood
import ood_detectors.sde as sde_lib 
import ood_detectors.models as models
import ood_detectors.losses as losses
...
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

ood_detector = likelihood.Likelihood(
    sde = sde,
    model = model,
    optimizer = optimizer,
    ).to(device)

update_fn = functools.partial(
    losses.SDE_EMA_Warmup_GradClip, 
    ema_rate=ema_rate,
    warmup=warmup,
    grad_clip=grad_clip,
    continuous=continuous,
    reduce_mean=reduce_mean,
    likelihood_weighting=likelihood_weighting,
    )

train_loss = ood_detector.fit(
    train_data,  
    n_epochs=n_epochs,
    batch_size=batch_size,
    update_fn=update_fn,
    )
```

## Create a custom component

You can create a custom component by doing the same thing as the library does. Good luck!


## Evaluate 

To assess the performance of the OOD detectors, you can utilize the following metrics:

- AUC: Area under the ROC curve
- FPR95: False positive rate when the true positive rate is 95%

```python
import ood_detectors.eval_utils as eval_utils
score_id = ood_detector.predict(train_data)
score_ref = ood_detector.predict(reference_data)
print(f"Train AUC: {eval_utils.auc(-score_ref, -score_id):.2%}")
print(f"Train FPR95: {eval_utils.fpr95(-score_ref, -score_id):.2%}")
```

```python
results = eval_utils.eval_ood(ood_detector, train_data, reference_data, ood_data, batch_size, verbose=False)
plot_utils.plot(results, id_name, ood_names, encoder=embedding, model=ood_detector.name,
                train_loss=train_loss,
                config=conf,
                )
```


## License

`ood_detectors` is distributed under the terms of the [apache-2.0](https://choosealicense.com/licenses/apache-2.0/) license.
