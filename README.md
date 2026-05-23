# Predicting-Landscape-Evolution-with-a-Deep-Learning-Surrogate



This project formulates landscape evolution prediction as a spatiotemporal forecasting problem. Given a sequence of past elevation maps, the model predicts future terrain states using a SimVP-based architecture from OpenSTL. The goal is not only to reduce pixel-level prediction error, but also to evaluate whether the learned surrogate preserves physically meaningful geomorphic structures such as ridge-valley spacing.

---

## Overview

Numerical landscape evolution models are computationally expensive, especially when running many simulations over different physical parameters and initial conditions. This project explores whether a neural surrogate model can learn the dominant temporal dynamics of simulated terrain evolution and produce reliable long-horizon forecasts.

The main model is trained in a **10-frame-to-10-frame** setting:

```text
Input : frames t      ... t+9
Output: frames t+10   ... t+19
```
After training, the model is evaluated in a long-horizon autoregressive rollout setting:
```text
Given frames 0–9,
predict frames 10–119.
```
A linear extrapolation baseline is used for comparison. The baseline estimates a pixel-wise linear trend from frames 0–9 and extrapolates that trend into the future.

## Repository Structure
The intended project layout is:
```text
project/
│
└── OpenSTL/
    │
    ├── all_z.npy
    ├── simvp_train_zonly_10to10.ipynb
    ├── simvp_verify.ipynb
    ├── test_pred_10-119.ipynb
    │
    ├── simvp_zonly_10to10_stride5/
    │   └── generated model checkpoints, predictions, metrics, and figures
    │
    └── README.md
```
The dataset file all_z.npy is intentionally not included in this repository because it is too large (384MB) for normal GitHub upload. Please place it manually under:
```text
OpenSTL/all_z.npy
```
or update the dataset path inside the notebooks.

## Notebooks
### `simvp_train_zonly_10to10.ipynb`

Main training notebook.

This notebook performs:
- environment and CUDA checks
- dataset loading
- train / validation / test split
- normalization using training simulations only
- SimVP model construction
- 10-to-10 training
- short-term test evaluation
- long-horizon autoregressive rollout
- linear extrapolation baseline comparison
- Fourier-based geomorphic evaluation
- result visualization and metric export

Default training setup:
```text
Input length     : 10 frames
Prediction length: 10 frames
Stride           : 5
Epochs           : 30
Loss             : MSE
Optimizer        : AdamW
Scheduler        : CosineAnnealingLR
Model            : SimVP
```
### `simvp_verify.ipynb`

Dataset and output verification notebook.

This notebook is used for checking:
- dataset shape
- saved .npy file shape
- prediction array consistency
- visualization sanity checks
- whether outputs follow the required format

Expected dataset format:
```text
all_z.shape == (50, 120, 128, 128)
```
with dimensions:
```text
50  = number of simulations
120 = number of frames per simulation
128 = grid height
128 = grid width
```
The required test output format is:
```text
test_z.shape == (10, 120, 128, 128)
```

### `test_pred_10-119.ipynb`
Long-horizon prediction and visualization notebook.

This notebook focuses on using frames 0–9 as input and evaluating predictions over frames 10–119.

It is mainly used for:

- loading trained model outputs
- visualizing long-rollout predictions
- comparing SimVP with the linear extrapolation baseline
- preparing figures for reports or presentations

## Dataset
steven!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


## OpenSTL Setup
This project depends on the SimVP implementation provided by [OpenSTL GitHub](https://github.com/chengtan9907/OpenSTL). The notebooks import SimVP using:
```text
from openstl.models import SimVP_Model
```
Therefore, OpenSTL must be available in the Python environment before running the training notebook.

1. Clone this project
```text
git clone <your-repository-url>
cd Predicting-Landscape-Evolution-with-a-Deep-Learning-Surrogate
```
2. Clone OpenSTL inside the project directory
```text
git clone https://github.com/chengtan9907/OpenSTL.git
```

3. Install OpenSTL in editable mode
From the project root:
```text
cd OpenSTL
pip install -e .
cd ..
```
After installation, verify that the import works:
```text
python -c "from openstl.models import SimVP_Model; print('OpenSTL SimVP import OK')"
```
If this command runs successfully, the notebooks should be able to use SimVP.

