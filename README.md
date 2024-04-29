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

- DDM_Likelihood: A likelihood-based detector that utilizes score-based denoising diffusion models (DDM) to map complex distributions to known ones.

- Residual: This method employs the least significant eigen vector for OOD detection.

All detectors share a common interface:

1. Initialize the detector with necessary hyperparameters.
2. Fit the model using fit() with the training data.
3. Use predict() to obtain OOD scores for new data samples.

## Example
```python
from ood_detectors import DDM_Likelihood

ood_detector = DDM_Likelihood()
ood_detector.fit(train_data, n_epochs, batch_size)
scores = ood_detector.predict(test_data, batch_size)
```

```python
from ood_detectors import Residual

ood_detector = Residual()
ood_detector.fit(train_data)
scores = ood_detector.predict(test_data)
```
## Evaluate 

To assess the performance of the OOD detectors, you can utilize the following metrics:

- AUC: Area under the ROC curve
- FPR95: False positive rate when the true positive rate is 95%

```python
import ood_detectors.eval_utils as eval_utils
score_id = model_residual.predict(train_data)
score_ref = model_residual.predict(reference_data)
print(f"Train AUC: {eval_utils.auc(-score_ref, -score_id):.2%}")
print(f"Train FPR95: {eval_utils.fpr95(-score_ref, -score_id):.2%}")
```

```python
results = eval_utils.eval_ood(model, train_data, reference_data, ood_data, batch_size, verbose=False)
plot_utils.plot(results, id_name, ood_names, encoder=embedding, model=model.__class__.__name__,
                train_loss=train_loss,
                config=conf,
                )
```


## License

`ood_detectors` is distributed under the terms of the [apache-2.0](https://choosealicense.com/licenses/apache-2.0/) license.
