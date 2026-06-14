# Centralized Learning Pipeline

This module implements a unified, task-driven model architecture for centralized machine learning experiments.
All models are determined by **task** and **model_config** only, ensuring consistency across centralized and federated settings.

## Core Models

The unified model system provides three core model types:

### 1. **LogisticRegression** (Classification Only)
- Binary classification using a single linear layer
- Outputs raw logits (suitable for `BCEWithLogitsLoss`)
- Usage: `get_model(input_dim, "classification", {"type": "logreg"})`

### 2. **LinearRegression** (Regression Only)
- Single linear layer for continuous predictions
- No activation function applied
- Usage: `get_model(input_dim, "regression", {"type": "linear"})`

### 3. **MLP** (Classification or Regression)
- Flexible multi-layer perceptron with configurable hidden layers
- ReLU activations with optional dropout
- Usage for classification: `get_model(input_dim, "classification", {"type": "mlp", "hidden_dims": [64, 32], "dropout": 0.1})`
- Usage for regression: `get_model(input_dim, "regression", {"type": "mlp", "hidden_dims": [64, 32], "dropout": 0.1})`

## Task-Model Constraints

| Task | Allowed Models |
|------|---|
| **classification** | `logreg`, `mlp` |
| **regression** | `linear`, `mlp` |

## Module Documentation

- `data.py`
    - `load_dataloaders_from_hf`: Load dataset from Hugging Face and return PyTorch DataLoaders.
    - `save_to_hf`: Save a pandas DataFrame to Hugging Face dataset hub.

- `evaluate.py`
    - `evaluate`: Evaluate a PyTorch model and optionally log metrics to W&B.

- `preprocess.py`
    - `add_cyclic_hour`: Convert hour feature into cyclic representation.
    - `to_tensors`: Convert numpy arrays into PyTorch tensors.

- `model.py` (NEW - Unified Architecture)
    - `LogisticRegression(input_dim)`: Binary classification model
    - `LinearRegression(input_dim)`: Regression model
    - `MLPClassifier(input_dim, hidden_dims, dropout)`: Flexible classification MLP
    - `MLPRegressor(input_dim, hidden_dims, dropout)`: Flexible regression MLP
    - `get_model(input_dim, task, model_config)`: **UNIFIED FACTORY FUNCTION** for all models

- `save.py`
    - `save_model_artifact`: Save a PyTorch model locally in models/<dataset_name>/ and optionally log to W&B.

- `train.py`
    - `train`: Train a PyTorch model with optional W&B logging.

- `visualize.py`
    - `plot_regression_results`: Plot Actual vs Predicted values and Residual distribution.

- `explainability.py`
    - `compute_shap`: Compute SHAP values for classification or regression models. Uses `shap.Explainer` with a sigmoid-wrapped predict function for classification (returns shape `(N, F)`), and `shap.DeepExplainer` for regression (returns shape `(N, F)`).
    - `save_results`: Save SHAP values as a `.npy` file to `results/<method>/<dataset_name>/`.
    - `plot_summary`: Generate and save a SHAP beeswarm summary plot showing feature importances with correct labels and colors to `results/<method>/<dataset_name>/shap_summary.png`.


## Example Usage

```python
from src.model import get_model
from src.data import load_dataloaders_from_hf
from src.train import train
from src.evaluate import evaluate

# Load data
train_loader, test_loader = load_dataloaders_from_hf(
    dataset_name="adult-income-census", 
    batch_size=32
)

# Create classification model
model = get_model(
    input_dim=30,
    task="classification",
    model_config={"type": "logreg"}
)

# Train
train(model, train_loader, epochs=10, task="classification")

# Evaluate
evaluate(model, test_loader, task="classification")
```

## Architecture Notes

- **NO dataset-specific models**: Old models like `BikeSharingModel`, `AdultIncomeCensusModel`, etc. have been removed
- **Task-driven design**: All 4 datasets now use the same 3 model types
- **Federated compatibility**: Identical `get_model()` factory ensures models work identically in federated learning
- **Unified initialization**: All clients receive the same model architecture
