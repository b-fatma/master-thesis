# Master's Thesis

Code base for master thesis experiments

**Repository:** [github.com/b-fatma/master-thesis](https://github.com/b-fatma/master-thesis)  
**Frameworks:** PyTorch · Flower · Weights & Biases · scikit-learn

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Features](#2-core-features)
3. [Repository Structure](#3-repository-structure)
4. [Datasets](#4-datasets)
5. [Models](#5-models)
6. [Installation & Setup](#6-installation--setup)
7. [Usage](#7-usage)
8. [Configuration](#8-configuration)
9. [Adversarial Attacks](#9-adversarial-attacks)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Technologies](#11-technologies)
12. [Detection methods](#12-detection-methods)

---

## 1. Overview

This repository implements a unified framework for comparing **centralized** and **federated learning** approaches across diverse machine learning tasks.

- **Centralized Learning**: All data available to a single server; standard supervised learning pipeline.
- **Federated Learning (FedAvg)**: Training distributed across multiple simulated clients using gradient averaging; data remains private on clients.

Both settings use identical **task-driven models** (not dataset-specific), ensuring clean experimental design. The framework supports both **classification** and **regression** tasks across 4 real-world datasets.

---

## 2. Core Features

### 2.1 Unified Task-Driven Architecture
- **Single source of truth for models**: `get_model()` factory function
- **Task-model constraints enforceded**:
  - Classification: `LogisticRegression` or `MLP`
  - Regression: `LinearRegression` or `MLP`
- **Identical models across all datasets and settings** (centralized & federated)
- **Configuration-driven**: Tasks and models specified via config, not code

### 2.2 Supported Tasks
- **Classification**: Binary classification with `BCEWithLogitsLoss`
- **Regression**: Continuous value prediction with `MSELoss`

### 2.3 Training Frameworks
- **Centralized**: PyTorch + Jupyter notebooks with optional W&B logging
- **Federated**: Flower (flwr) with client-server architecture + FedAvg aggregation

### 2.4 Input/Output Flexibility
- **Data source**: Hugging Face datasets (automatic download & caching)
- **Model persistence**: Save/load via PyTorch state dicts
- **Experiment tracking**: Weights & Biases (optional offline mode)
- **Results artifacts**: Per-dataset model checkpoints + final aggregated model

---

## 3. Repository Structure

```
master-thesis/
├── requirements.txt                    # Python dependencies
│
├── centralized/                        # Centralized learning experiments
│   ├── models/                         # Saved trained models (per dataset)
│   │   ├── adult-income-census/
│   │   ├── bank-marketing/
│   │   ├── bike-sharing/
│   │   └── housing-prices/
│   ├── notebooks/                      # Jupyter notebooks (EDA → preprocessing → training)
│   │   ├── adult-income-census/
│   │   ├── bank-marketing/
│   │   ├── bike-sharing/
│   │   └── housing-price/
│   ├── attacks/                        # Adversarial attack experiments
│   │   ├── experiments/                # Attack scripts (label flip, etc.)
│   │   └── utils/                      # Attack utilities (logging, visualization)
│   └── src/                            # Centralized pipeline utilities
│       ├── config.py                   # Dataset configuration
│       ├── data.py                     # Data loading
│       ├── model.py                    # Unified model definitions
│       ├── train.py                    # Training loop
│       ├── evaluate.py                 # Evaluation & metrics
│       ├── save.py                     # Model checkpoint saving
│       ├── visualize.py                # Visualization (plots, ROC, confusion matrix)
│       ├── preprocess.py               # Feature engineering utilities
│       ├── utils.py                    # Common utilities (seeding, accuracy)
│       └── README.md                   # Centralized module documentation
│
├── federated/                          # Federated learning (Flower + FedAvg)
│   ├── final_model.pt                  # Final aggregated model checkpoint
│   ├── pyproject.toml                  # Flower config: hyperparameters, datasets, attacks
│   ├── pytorchexample/                 # Flower application
│   │   ├── client_app.py               # Client: local training & evaluation
│   │   ├── server_app.py               # Server: aggregation & global evaluation
│   │   ├── config.py                   # Config: datasets + attack settings
│   │   ├── models.py                   # Models: unified factory (logreg, linear, mlp)
│   │   ├── dataset.py                  # Dataset loading: HF + IID partitioning
│   │   ├── task.py                     # Training/evaluation routines (task-specific)
│   │   └── attacks.py                  # Attack dispatcher for malicious clients
│   └── wandb/                          # Weights & Biases experiment logs
│
└── src/                                # Shared attack implementations
    └── attacks/                        # Poisoning attacks (centralized & federated)
        ├── label_flip.py               # Label flip attack
        └── __init__.py                 # Attack dispatcher
```

---

## 4. Datasets

Four real-world datasets covering two task types (classification & regression):

| Dataset | Task | Instances | Features | HuggingFace Repo |
|---|---|---|---|---|
| **Adult Income Census** | Classification | ~26K (train) / ~6.5K (test) | 30 | `b-fatma/adult-income-census-federated` |
| **Bank Marketing** | Classification | ~44K (train) / ~8K (test) | 20 | `narimanee/bank-marketing-federated` |
| **Bike Sharing** | Regression | ~14K (train) / ~3.5K (test) | 12 | `b-fatma/bike-sharing-federated` |
| **Housing Prices** | Regression | ~16K (train) / ~4K (test) | 18 | `narimanee/Housing-prices-federated` |

**Data Pipeline**: All raw datasets are loaded in `centralized/notebooks/<dataset-name>/01_eda.ipynb` and `02_preprocessing.ipynb`, preprocessed (scaling, feature engineering, missing value handling), and then **uploaded to HuggingFace** via the `save_to_hf()` utility in `centralized/src/data.py`. These preprocessed datasets on HuggingFace are the canonical source for all centralized and federated training experiments.

All datasets are then:
- Loaded directly from HuggingFace for centralized training
- Automatically partitioned (IID) for federated experiments via `flwr_datasets`

---

## 5. Models

All models are **unified** (task-driven, not dataset-specific) and work identically across centralized and federated settings.

### Model Types

| Model | Task(s) | Description | Output |
|---|---|---|---|
| **LogisticRegression** | Classification | Single linear layer | Raw logits (for BCEWithLogitsLoss) |
| **LinearRegression** | Regression | Single linear layer | Continuous values |
| **MLPClassifier** | Classification | Multi-layer with ReLU + dropout | Raw logits |
| **MLPRegressor** | Regression | Multi-layer with ReLU + dropout | Continuous values |

### Usage

```python
from src.model import get_model

# Logistic regression (classification)
model = get_model(input_dim=30, task="classification", model_config={"type": "logreg"})

# Linear regression
model = get_model(input_dim=12, task="regression", model_config={"type": "linear"})

# MLP classifier with hidden layers
model = get_model(
    input_dim=30,
    task="classification",
    model_config={"type": "mlp", "hidden_dims": [64, 32], "dropout": 0.1}
)

# MLP regressor
model = get_model(
    input_dim=12,
    task="regression",
    model_config={"type": "mlp", "hidden_dims": [128, 64], "dropout": 0.2}
)
```

### Task-Model Constraints

```
Classification:     ✓ LogisticRegression  ✓ MLP
Regression:         ✓ LinearRegression    ✓ MLP
```

---

## 6. Installation & Setup

### Prerequisites

- Python 3.8+
- `pip` package manager
- Git

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/b-fatma/master-thesis.git
cd master-thesis
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. Install the Flower federated learning framework** *(required for federated experiments)*

```bash
cd federated
pip install -e .
```

---

## 7. Usage

### 7.1 Centralized Learning

Navigate to `centralized/notebooks/<dataset-name>/` and run the notebooks in sequence:

| Step | Notebook | Description |
|---|---|---|
| 1 | `01_eda.ipynb` | Exploratory Data Analysis — distributions, class balance, feature correlations |
| 2 | `02_preprocessing.ipynb` | Feature engineering, scaling, train/test splitting |
| 3 | `03_training.ipynb` | Model training with task-specific loss, evaluation, and artifact saving |

**Example:**

```bash
cd centralized/notebooks/adult-income-census
# Run 01_eda.ipynb → 02_preprocessing.ipynb → 03_training.ipynb in Jupyter
```

### 7.2 Federated Learning

**Step 1 — Configure the federated experiment** (`federated/pyproject.toml`):

```toml
[tool.flwr.app.config]
dataset-name = "adult-income-census"  # Options: "adult-income-census", "bike-sharing", 
                                      #          "bank-marketing", "housing-prices"
num-server-rounds = 10
local-epochs = 2
learning-rate = 0.01
batch-size = 8
```

**Step 2 — Run the federated simulation:**

```bash
cd federated
flwr run
```

The server-side global evaluation occurs at the end of each round. Final aggregated model is saved to `final_model.pt`.

---

## 8. Configuration

### 8.1 Centralized Learning

Configuration is handled per-notebook in `centralized/notebooks/<dataset>/03_training.ipynb`:

```python
from src.config import get_dataset_config

config = get_dataset_config("adult-income-census")
# Returns: DatasetConfig with task, hf_repo, label_col, input_dim, model_config
```

All dataset configurations are centralized in:
- **Centralized**: `centralized/src/config.py`
- **Federated**: `federated/pytorchexample/config.py`

Each `DatasetConfig` specifies:
- `name`: Dataset identifier
- `task`: "classification" or "regression"
- `hf_repo`: HuggingFace repository
- `label_col`: Target column name
- `input_dim`: Number of features
- `model_config`: Model type & hyperparameters

### 8.2 Federated Learning

All hyperparameters are managed in `federated/pyproject.toml` under `[tool.flwr.app.config]`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset-name` | string | `"adult-income-census"` | Dataset to use |
| `num-server-rounds` | int | `10` | Total federated training rounds |
| `fraction-evaluate` | float | `1.0` | Fraction of clients participating in evaluation per round |
| `local-epochs` | int | `2` | Local training epochs on each client per round |
| `learning-rate` | float | `0.01` | Adam optimizer learning rate |
| `batch-size` | int | `8` | Local batch size during client training |

**System-level settings** *(optional, configured in `$HOME/.flwr/config.toml`)*:

```toml
[options]
num-supernodes = 10                                      # Number of CPU cores for simulation
backend = "simulation"                                  # "simulation" or "simulation-gpu"
backend.client-resources.num-cpus = 4                   # CPUs per simulated client
backend.client-resources.num-gpus = 0                   # GPUs per simulated client
```

---

## 9. Adversarial Attacks

This project supports adversarial attacks in both **centralized** and **federated learning** settings. The attacks are divided into two main categories:

* **Data Poisoning Attacks**: Manipulate training data or labels
* **Model Poisoning Attacks**: Manipulate model updates before aggregation

---

### 9.1. Data Poisoning Attacks

Data poisoning attacks corrupt the training dataset at the client level before or during training.

#### Overview

These attacks are applied **locally on each malicious client’s dataset** and are compatible with both:

* Centralized training (standalone experiments)
* Federated learning (distributed setting)

---

#### 1. Label Flip Attack (Classification)

Malicious clients flip labels from a **victim class** to a **target class**, degrading classification performance.

##### Mechanism

Given a label ( y ), the poisoned label becomes:

[
y_{\text{poison}} =
\begin{cases}
\text{target-label} & \text{if } y = \text{victim-label} \
y & \text{otherwise}
\end{cases}
]

---

##### Centralized Usage

```bash
cd centralized/attacks/experiments
python run_label_flip.py
python run_label_flip.py --wandb
```

---

#### Federated Configuration

```toml
[tool.flwr.app.config]
attack-enabled = true
attack-type = "label_flip"

malicious-ratio = 0.2
poison-fraction = 0.6
victim-label = 1
target-label = 0
attack-seed = 42
```

---

##### Parameters

| Parameter         | Description                         |
| ----------------- | ----------------------------------- |
| `poison-fraction` | Fraction of local samples to poison |
| `victim-label`    | Label to flip from                  |
| `target-label`    | Label to flip to                    |
| `malicious-ratio` | Fraction of malicious clients       |
| `attack-seed`     | Ensures reproducibility             |

---

#### 2. Distribution Shift Attack (Regression)

Malicious clients corrupt continuous targets by injecting noise.

---

##### Mechanisms

###### Gaussian Noise

[
y_{\text{poison}} = y + \mathcal{N}(0, \sigma^2)
]

###### Uniform Noise

[
y_{\text{poison}} = y + \mathcal{U}(-a, a)
]

---

##### Victim Selection Strategies

| Strategy         | Description                  |
| ---------------- | ---------------------------- |
| `random`         | Random subset of data        |
| `quantile_range` | Targets within a value range |

---

##### Centralized Usage

```bash
cd centralized/attacks/experiments
python run_distribution_shift.py
python run_distribution_shift.py --wandb
```

---

##### Federated Configuration

```toml
[tool.flwr.app.config]
attack-enabled = true
attack-type = "distribution_shift"

shift-mechanism = "noise_gaussian"
shift-param = 5.0

selection-strategy = "random"
victim-quantile-min = 0.0
victim-quantile-max = 1.0

poison-fraction = 0.3
malicious-ratio = 0.2
attack-seed = 42
```

---

##### Parameters

| Parameter            | Description                                    |
| -------------------- | ---------------------------------------------- |
| `shift-mechanism`    | Noise type (`noise_gaussian`, `noise_uniform`) |
| `shift-param`        | Noise magnitude                                |
| `selection-strategy` | Data selection method                          |
| `poison-fraction`    | Fraction of targets to corrupt                 |
| `malicious-ratio`    | Fraction of malicious clients                  |
| `attack-seed`        | Reproducibility seed                           |

---

### 9.2. Model Poisoning Attacks

Model poisoning attacks manipulate **client model updates** after local training and before aggregation.

#### Overview

* Applied **after local training**
* Modify uploaded model parameters
* Fully compatible with federated learning pipelines

---

#### Supported Attack Types

* `history`
* `mpaf`

---

#### 1. History Attack

Malicious clients amplify their update direction relative to the global model.

##### Formula

\theta_{\mathrm{poisoned}} = \theta_{\mathrm{ref}} + \lambda \left(\theta_{\mathrm{local}} - \theta_{\mathrm{ref}}\right)

##### Intuition

* ( \theta_{\text{local}} - \theta_{\text{ref}} ): local update
* ( \lambda > 1 ): amplifies the update
* Pushes the global model more aggressively in the malicious direction

---

#### 2. MPAF Attack

MPAF (Model Poisoning with Adversarial Feedback) uses a randomized base model.

##### Formula

\theta_{\mathrm{poisoned}} = \theta_{\mathrm{ref}} + \lambda \left(\theta_{\mathrm{base}} - \theta_{\mathrm{local}}\right)

---

##### Key Characteristics

* Uses a deterministic base model ( \theta_{\text{base}} )
* Base model is generated using:

  * Seeded random generator
  * Client partition ID
* Ensures reproducibility across runs

---

##### Base Model Initialization

* Each parameter tensor is sampled from:

[
\mathcal{N}(0, 1)
]

* Seeded per malicious client

---

#### Federated Configuration

```toml
[tool.flwr.app.config]
attack-enabled = true
attack-type = "history"   # or "mpaf"

model-attack-lambda = 2.0
malicious-ratio = 0.2
attack-seed = 42
```

---

##### Parameters

| Parameter             | Description                        |
| --------------------- | ---------------------------------- |
| `attack-type`         | `history` or `mpaf`                |
| `model-attack-lambda` | Scaling factor for attack strength |
| `malicious-ratio`     | Fraction of malicious clients      |
| `attack-seed`         | Reproducibility seed               |

---

### Attack Compatibility Summary

| Attack             | Category        | Task Type      | Description                   | Status |
| ------------------ | --------------- | -------------- | ----------------------------- | ------ |
| Label Flip         | Data Poisoning  | Classification | Label corruption              | ✓      |
| Distribution Shift | Data Poisoning  | Regression     | Target noise injection        | ✓      |
| History            | Model Poisoning | Any            | Amplified update direction    | ✓      |
| MPAF               | Model Poisoning | Any            | Randomized adversarial update | ✓      |

---

### Notes

* Malicious clients are selected **deterministically** using `attack-seed`
* All attacks are **modular** and configurable via `pyproject.toml`
* Model poisoning attacks operate at the **parameter level**, while data poisoning attacks operate at the **dataset level**

---

## 12. Detection methods

This repository includes multiple, distinct server-side detection approaches. They are implemented in `federated/src/detection/` and used independently in experiments and plotting pipelines.

- **FLDetector** — a windowed update-distance detector adapted from FLPoison. It analyzes a sliding window of recent global update distances, applies gap statistics and KMeans clustering to separate benign and suspicious clients, and filters flagged clients before aggregation. Implementation: `federated/src/detection/fldetector.py`.

- **SHAPCOSIM (mad-shapcosim)** — a lightweight hybrid detector that z-scores server-reference and mean pairwise cosine signals, fuses them with weight `alpha`, and applies a one-sided MAD threshold (median - k * 1.4826 * MAD) to flag unusually low-scoring clients. Implementation: `federated/src/detection/cosine.py`.


---

## 10. Evaluation Metrics

### Classification Tasks

The framework computes the following metrics:

| Metric | Definition | Task |
|---|---|---|
| **Accuracy** | Fraction of correct predictions | Binary classification |
| **Precision** | True positives / (true positives + false positives) | Binary classification |
| **Recall** | True positives / (true positives + false negatives) | Binary classification |
| **F1-Score** | Harmonic mean of precision and recall | Binary classification |

### Regression Tasks

| Metric | Definition | Task |
|---|---|---|
| **MSE** | Mean squared error | Continuous prediction |
| **RMSE** | √MSE | Continuous prediction |
| **R²** | Coefficient of determination (0–1, higher is better) | Continuous prediction |

### Federated Results

- Metrics are computed **per round** during training
- Server evaluates the global model on the full test set
- Client metrics are averaged across all participating clients
- All results are tracked in `federated/wandb/` and plotted for comparison

---

## 11. Technologies

| Library / Tool | Role |
|---|---|
| **PyTorch** | Core deep learning framework for model definition and training |
| **Flower (flwr)** | Federated learning orchestration with FedAvg aggregation |
| **scikit-learn** | Metrics computation, data scaling, train/test splitting |
| **Datasets (HuggingFace)** | Dataset loading and distribution across clients |
| **pandas / numpy** | Data manipulation and numerical operations |
| **matplotlib / seaborn** | Visualization for EDA and result analysis |
| **Jupyter** | Interactive notebooks for preprocessing and analysis |
| **Weights & Biases (wandb)** | Experiment tracking, per-round metrics, and run comparison |

---

## Additional Resources

- **Centralized Pipeline**: See `centralized/src/README.md` for detailed module documentation
- **Model Unified Factory**: `src/model.py` and `federated/pytorchexample/models.py` (identical implementations)
- **Attacks Implementation**: `src/attacks/label_flip.py` for shared attack logic
- **Dataset Configurations**: `centralized/src/config.py` and `federated/pytorchexample/config.py`
- **Flower Documentation**: https://flower.ai/docs
