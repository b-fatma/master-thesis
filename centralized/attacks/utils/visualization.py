"""
Visualization utilities for attack experiments.

One function per visualization type, all save to results/plots/.
Call these at the end of each experiment script.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PLOTS_DIR = Path("./results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_accuracy_over_rounds(
    rounds,
    clean_accs,
    attacked_accs,
    attack_name,
    save=True,
):
    """
    Line plot comparing clean vs attacked model accuracy over FL rounds.
    Use for poisoning + Byzantine experiments.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rounds, clean_accs, "b-o", label="Clean model", linewidth=2)
    ax.plot(
        rounds,
        attacked_accs,
        "r-x",
        label=f"Under {attack_name}",
        linewidth=2,
        linestyle="--",
    )
    ax.fill_between(rounds, clean_accs, attacked_accs, alpha=0.1, color="red")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy Degradation — {attack_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f"{attack_name}_accuracy_rounds.png"
        fig.savefig(path, dpi=150)
        print(f"[viz] Saved → {path}")
    return fig


def plot_poison_fraction_sweep(
    fractions,
    clean_accs,
    attacked_accs,
    asrs=None,
    attack_name="label_flip",
    save=True,
):
    """
    Show how accuracy and ASR change as poison fraction increases.
    Key plot for understanding attack budget vs effect trade-off.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(fractions, clean_accs, "b-o", label="Clean accuracy")
    ax1.plot(fractions, attacked_accs, "r-s", label="Poisoned accuracy")
    ax1.set_xlabel("Poison fraction")
    ax1.set_ylabel("Accuracy", color="black")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    if asrs is not None:
        ax2 = ax1.twinx()
        ax2.plot(fractions, asrs, "g--^", label="ASR")
        ax2.set_ylabel("Attack Success Rate", color="green")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor="green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    if asrs is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    else:
        ax1.legend()

    ax1.set_title(f"Poison Fraction vs Effect — {attack_name}")
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f"{attack_name}_fraction_sweep.png"
        fig.savefig(path, dpi=150)
        print(f"[viz] Saved → {path}")
    return fig


def plot_decision_boundary(
    model_clean,
    model_attacked,
    X_test,
    y_test,
    attack_name="label_flip",
    save=True,
):
    """
    2D decision boundary comparison: clean vs attacked model.
    Only works with 2-feature datasets (use synthetic 2D loader).
    """
    h = 0.02
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, title in zip(
        axes, [model_clean, model_attacked], ["Clean model", f"After {attack_name}"]
    ):
        Z = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=plt.cm.RdYlBu,
            edgecolors="k",
            s=20,
        )
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    plt.suptitle(f"Decision Boundary Shift — {attack_name}", fontsize=13)
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f"{attack_name}_decision_boundary.png"
        fig.savefig(path, dpi=150)
        print(f"[viz] Saved → {path}")
    return fig


def plot_gradient_norms(
    client_ids,
    grad_norms,
    malicious_ids=None,
    round_num=None,
    attack_name="byzantine",
    save=True,
):
    """
    Bar chart of per-client gradient norms.
    Malicious clients typically have anomalous norms.
    This is the XAI bridge — detecting attacks via gradient statistics.
    """
    colors = []
    for cid in client_ids:
        if malicious_ids and cid in malicious_ids:
            colors.append("red")
        else:
            colors.append("steelblue")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(c) for c in client_ids], grad_norms, color=colors)
    ax.axhline(np.mean(grad_norms), color="orange", linestyle="--", label="Mean norm")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Gradient L2 Norm")
    title = f"Per-Client Gradient Norms — {attack_name}"
    if round_num is not None:
        title += f" (Round {round_num})"
    ax.set_title(title)
    ax.legend()

    # legend patch for malicious
    if malicious_ids:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", label="Malicious client"),
            Patch(facecolor="steelblue", label="Benign client"),
        ]
        ax.legend(handles=legend_elements)

    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f"{attack_name}_grad_norms_r{round_num}.png"
        fig.savefig(path, dpi=150)
        print(f"[viz] Saved → {path}")
    return fig


def plot_reconstructed_images(
    originals, reconstructions, attack_name="dlg", n=5, save=True
):
    """
    Side-by-side: original vs gradient-inverted reconstruction.
    For DLG / GradInversion experiments.
    """
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
    for i in range(n):
        axes[0, i].imshow(np.array(originals[i]).squeeze(), cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        axes[1, i].imshow(np.array(reconstructions[i]).squeeze(), cmap="gray")
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")
    plt.suptitle(f"Gradient Inversion — {attack_name}", fontsize=13)
    plt.tight_layout()
    if save:
        path = PLOTS_DIR / f"{attack_name}_reconstructions.png"
        fig.savefig(path, dpi=150)
        print(f"[viz] Saved → {path}")
    return fig


def plot_distribution_shift_sweep(
    results,
    clean_baseline,
    shift_mechanism="noise_gaussian",
    shift_params=None,
    save=True,
):
    """
    Generate 2 figures for distribution shift attack hyperparameter sweep.
    Works with any shift mechanism (Gaussian noise, uniform noise, additive, etc).

    Figure 1: Absolute Metrics vs Shift Parameter
      - 4 subplots (MSE, MAE, R², Bias) in 2×2 layout
      - X-axis: shift_param values (noise_std, noise_bound, delta, etc.)
      - Lines: one per poison_frac
      - Reference: horizontal clean baseline
      - Shows: raw model performance degradation

    Figure 2: Degradation Metrics vs Shift Parameter
      - 4 subplots (MSE increase, R² decrease, MAE increase, Bias shift)
      - X-axis: shift_param values
      - Lines: one per poison_frac
      - Y-axis: starts at 0 (clean baseline = no degradation)
      - Shows: relative attack impact

    Args:
        results (list): List of result dicts from experiment
            Each dict contains:
            - shift_param (float): The shift parameter value
            - poison_frac (float): Fraction of samples poisoned
            - clean_metrics (dict: 'mse', 'mae', 'r_squared', 'bias')
            - attacked_metrics (dict: same keys)

        clean_baseline (dict): Clean model metrics
            Keys: 'mse', 'mae', 'r_squared', 'bias'

        shift_mechanism (str): Type of shift mechanism
            Examples: "noise_gaussian", "noise_uniform", "additive", etc.

        shift_params (list): List of shift parameter values (for sorting x-axis).
            If None, extracted and sorted from results.

        save (bool): Whether to save figures to PNG

    Returns:
        tuple: (fig1, fig2) matplotlib figure objects
    """
    # Organize results by shift_param and poison_frac
    param_values = (
        sorted(set(r["shift_param"] for r in results))
        if shift_params is None
        else shift_params
    )
    poison_fracs = sorted(set(r["poison_frac"] for r in results))

    # Map mechanism name to display name
    mechanism_display = {
        "noise_gaussian": "Gaussian Noise (σ)",
        "noise_uniform": "Uniform Noise (±a)",
        "additive": "Additive Shift (Δ)",
        "multiplicative": "Multiplicative Shift (scale)",
    }
    param_label = mechanism_display.get(shift_mechanism, shift_mechanism)

    # Create lookup: (shift_param, poison_frac) -> result
    results_lookup = {(r["shift_param"], r["poison_frac"]): r for r in results}

    # Color map for poison fractions
    colors = plt.cm.Set2(np.linspace(0, 1, len(poison_fracs)))
    poison_frac_colors = {frac: colors[i] for i, frac in enumerate(poison_fracs)}

    # ──────────────────────────────────────────────────────────────
    # Figure 1: Absolute Metrics
    # ──────────────────────────────────────────────────────────────

    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))
    axes1 = axes1.flatten()

    metric_names = ["mse", "mae", "r_squared", "bias"]
    metric_labels = ["MSE", "MAE", "R²", "Bias"]

    for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes1[idx]

        # Plot clean baseline
        clean_val = clean_baseline[metric_name]
        ax.axhline(
            clean_val,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Clean baseline",
        )

        # Plot lines for each poison fraction
        for poison_frac in poison_fracs:
            attacked_vals = []
            for param_val in param_values:
                result = results_lookup.get((param_val, poison_frac))
                if result:
                    attacked_vals.append(result["attacked_metrics"][metric_name])
                else:
                    attacked_vals.append(np.nan)

            ax.plot(
                param_values,
                attacked_vals,
                marker="o",
                linewidth=2,
                label=f"Poison frac = {poison_frac:.0%}",
                color=poison_frac_colors[poison_frac],
            )

        ax.set_xlabel(param_label, fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(
            f"Absolute {metric_label} vs {param_label}", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig1.suptitle(
        f"Distribution Shift Attack ({shift_mechanism}) — Absolute Metrics",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / f"{shift_mechanism}_absolute_metrics.png"
        fig1.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved → {path}")

    # ──────────────────────────────────────────────────────────────
    # Figure 2: Degradation Metrics
    # ──────────────────────────────────────────────────────────────

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    axes2 = axes2.flatten()

    degradation_keys = ["mse_increase", "r2_decrease", "mae_increase", "bias_shift"]
    degradation_labels = ["MSE Increase", "R² Decrease", "MAE Increase", "Bias Shift"]

    for idx, (deg_key, deg_label) in enumerate(
        zip(degradation_keys, degradation_labels)
    ):
        ax = axes2[idx]

        # Plot clean baseline (y=0, no degradation)
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Clean baseline (no degradation)",
        )

        # Plot lines for each poison fraction
        for poison_frac in poison_fracs:
            degradation_vals = []
            for param_val in param_values:
                result = results_lookup.get((param_val, poison_frac))
                if result:
                    if deg_key == "mse_increase":
                        deg_val = (
                            result["attacked_metrics"]["mse"] - clean_baseline["mse"]
                        )
                    elif deg_key == "r2_decrease":
                        deg_val = (
                            clean_baseline["r_squared"]
                            - result["attacked_metrics"]["r_squared"]
                        )
                    elif deg_key == "mae_increase":
                        deg_val = (
                            result["attacked_metrics"]["mae"] - clean_baseline["mae"]
                        )
                    elif deg_key == "bias_shift":
                        deg_val = (
                            result["attacked_metrics"]["bias"] - clean_baseline["bias"]
                        )
                    degradation_vals.append(deg_val)
                else:
                    degradation_vals.append(np.nan)

            ax.plot(
                param_values,
                degradation_vals,
                marker="s",
                linewidth=2,
                label=f"Poison frac = {poison_frac:.0%}",
                color=poison_frac_colors[poison_frac],
            )

        ax.set_xlabel(param_label, fontsize=11)
        ax.set_ylabel(f"Δ {deg_label}", fontsize=11)
        ax.set_title(f"{deg_label} vs {param_label}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    fig2.suptitle(
        f"Distribution Shift Attack ({shift_mechanism}) — Metric Degradation",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save:
        path = PLOTS_DIR / f"{shift_mechanism}_degradation_metrics.png"
        fig2.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[viz] Saved → {path}")

    return fig1, fig2


# Backward compatibility alias
def plot_gaussian_noise_sweep(
    results, clean_baseline, attack_name="gaussian_noise", save=True
):
    """Deprecated: Use plot_distribution_shift_sweep() instead."""
    return plot_distribution_shift_sweep(
        results=results,
        clean_baseline=clean_baseline,
        shift_mechanism="noise_gaussian",
        save=save,
    )
