"""
Visualization utilities for evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_regression_results(
    y_true, y_pred, title="Regression Results", figsize=(12, 5)
):
    """
    Plot Actual vs Predicted values and Residual distribution.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values from the model.
        title (str, optional): Main title for the plots. Defaults to "Regression Results".
        figsize (tuple, optional): Figure size. Defaults to (12,5).

    Returns:
        None: Shows the plots.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Scatter plot: Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    axes[0].plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
    )
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Actual vs Predicted")

    # Residual distribution
    sns.histplot(residuals, bins=30, kde=True, ax=axes[1])
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", figsize=(8, 6)):
    """
    Plot confusion matrix for binary classification.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values from the model.
        title (str, optional): Title for the plot. Defaults to "Confusion Matrix".
        figsize (tuple, optional): Figure size. Defaults to (8,6).

    Returns:
        None: Shows the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_probs, title="ROC Curve", figsize=(8, 6)):
    """
    Plot ROC curve for binary classification.

    Args:
        y_true (array-like): True target values.
        y_probs (array-like): Predicted probabilities from the model.
        title (str, optional): Title for the plot. Defaults to "ROC Curve".
        figsize (tuple, optional): Figure size. Defaults to (8,6).

    Returns:
        None: Shows the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
