import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt


DATASETS = ["adult-income-census", "bank-marketing", "bike-sharing", "housing-prices"]


def compute_shap(model, X_train, X_test, task="classification"):
    """
    Computes SHAP values for both classification and regression models.

    Classification (adult-income-census):
        Uses shap.Explainer with a sigmoid-wrapped predict function.
        Returns shape (N, F) — probabilities of the positive class.

    Regression (bank-marketing, bike-sharing, housing-prices):
        Uses DeepExplainer directly on the raw model output.
        DeepExplainer on a regression model (out_features=1) returns
        a list containing one array of shape (N, F) — we extract [0]
        and squeeze any trailing dim-1 axes to guarantee (N, F).

    Args:
        model:   Trained nn.Module in eval mode.
        X_train: np.ndarray or torch.Tensor, shape (N_train, F).
        X_test:  np.ndarray or torch.Tensor, shape (N_test, F).
        task:    "classification" or "regression".

    Returns:
        shap_values: np.ndarray of shape (N_test, F).
    """
    model.eval()

    # ── Normalise inputs to the right type per explainer ──────────────────────
    if isinstance(X_train, np.ndarray):
        X_train_tensor = torch.from_numpy(X_train).float()
        X_train_np = X_train.astype(np.float32)
    else:
        X_train_tensor = X_train.float()
        X_train_np = X_train.detach().numpy().astype(np.float32)

    if isinstance(X_test, np.ndarray):
        X_test_tensor = torch.from_numpy(X_test).float()
        X_test_np = X_test.astype(np.float32)
    else:
        X_test_tensor = X_test.float()
        X_test_np = X_test.detach().numpy().astype(np.float32)

    background = X_train_tensor[:100]

    # ── Classification ─────────────────────────────────────────────────────────
    if task == "classification":
        # Wrap model so it outputs a probability scalar per sample.
        # shap.Explainer expects a callable: np.ndarray → np.ndarray (N,)
        def model_predict(x: np.ndarray) -> np.ndarray:
            t = torch.from_numpy(x).float()
            with torch.no_grad():
                prob = torch.sigmoid(model(t))  # (N, 1)
            return prob.numpy().squeeze(-1)  # (N,)

        explainer = shap.Explainer(model_predict, X_train_np[:100])
        shap_obj = explainer(X_test_np)  # shap.Explanation object
        shap_values = shap_obj.values  # (N, F)

    # ── Regression ─────────────────────────────────────────────────────────────
    else:
        explainer = shap.DeepExplainer(model, background)
        raw = explainer.shap_values(X_test_tensor)

        # DeepExplainer on a regression model (out_features=1) returns either:
        #   • list of 1 array  → raw[0] shape (N, F) or (N, F, 1)
        #   • single ndarray   → shape (N, F) or (N, F, 1)
        # We extract and squeeze all trailing size-1 dims → guaranteed (N, F)
        if isinstance(raw, list):
            shap_values = raw[0]
        else:
            shap_values = raw

        shap_values = np.squeeze(shap_values)

        # Edge case: single test sample → squeeze gives (F,), restore to (1, F)
        if shap_values.ndim == 1:
            shap_values = shap_values[np.newaxis, :]

    return shap_values  # np.ndarray, always (N, F)


def save_results(shap_values, dataset_name, method="shap"):
    output_dir = f"results/{method}/{dataset_name}/"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "shap_values.npy"), shap_values)
    print(f"[SHAP] Values saved → {output_dir}shap_values.npy")


def plot_summary(shap_values, X_test, feature_names, dataset_name, method="shap"):
    """
    Beeswarm summary plot showing all features with correct labels and colors.

    Args:
        shap_values:   np.ndarray (N, F).
        X_test:        np.ndarray or tensor (N, F) — used for dot coloring.
        feature_names: list of F strings — was missing before, caused the
                       single-feature / axis-label bug in the original plots.
        dataset_name:  used for the output path and plot title.
        method:        subfolder under results/ (default "shap").
    """
    output_dir = f"results/{method}/{dataset_name}/"
    os.makedirs(output_dir, exist_ok=True)

    if torch.is_tensor(X_test):
        X_test = X_test.detach().numpy()

    # Catch shape mismatches early with a clear message
    assert shap_values.shape == X_test.shape, (
        f"Shape mismatch: shap_values {shap_values.shape} vs X_test {X_test.shape}"
    )
    assert len(feature_names) == shap_values.shape[1], (
        f"feature_names length {len(feature_names)} does not match "
        f"shap_values n_features {shap_values.shape[1]}"
    )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,  # ← this was missing — caused the bug
        show=False,
    )
    plt.title(f"SHAP Summary — {dataset_name}", fontsize=12, pad=12)
    plt.savefig(
        os.path.join(output_dir, "shap_summary.png"),
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()
    print(f"[SHAP] Summary plot saved → {output_dir}shap_summary.png")
