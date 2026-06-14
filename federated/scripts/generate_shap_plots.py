"""Generate SHAP plots from saved .npz artifacts.

This script is an offline utility for regenerating SHAP plots from existing
artifacts saved by the federated learning server. It does not create a new
Weights & Biases run and it does not upload artifacts.

Usage:

    conda activate flwr && python scripts/generate_shap_plots.py

The script will:
- Find the latest run directory under ``experiments/shap_outputs/``
- Load all ``.npz`` artifacts for that run
- Generate bar and beeswarm plots using the plotting helpers

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the local packages are importable when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Save SHAP outputs under the research-friendly experiments directory
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "shap_outputs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from shap_pipeline.loading import load_local_shap_artifacts
    from shap_pipeline.plotting import generate_shap_plots
except Exception:
    # Fall back to the package-style import when the project is installed as src.
    from src.shap_pipeline.loading import load_local_shap_artifacts
    from src.shap_pipeline.plotting import generate_shap_plots

logging.basicConfig(level=logging.INFO)


def find_latest_run_dir(root: str) -> Path | None:
    """Return the most recently modified run directory under a root path.

    Args:
        root: Directory containing one subdirectory per federated run.

    Returns:
        The newest run directory, or ``None`` if no run directories exist.
    """
    root_path = Path(root)
    if not root_path.exists():
        return None
    runs = [directory for directory in root_path.iterdir() if directory.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def main(output_root: str = str(DEFAULT_OUTPUT_ROOT)) -> int:
    """Regenerate plots for the latest saved SHAP run.

    Args:
        output_root: Root directory containing federated SHAP outputs.

    Returns:
        Process exit code. Returns ``0`` on success, ``2`` when no artifacts are found.
    """
    run_dir = find_latest_run_dir(output_root)
    if run_dir is None:
        logging.error("No run directories found under %s", output_root)
        return 2

    logging.info("Using run directory: %s", run_dir)
    artifacts = load_local_shap_artifacts(str(run_dir))
    if not artifacts:
        logging.error("No SHAP artifacts found under %s", run_dir)
        return 2

    saved_paths = []
    for artifact in artifacts:
        logging.info("Generating plots for: %s", artifact.values_path)
        plots = generate_shap_plots(
            shap_values=artifact.values,
            feature_names=artifact.feature_names,
            metadata=artifact.metadata,
            output_root=output_root,
            explanation_data=artifact.explanation_data,
        )
        saved_paths.append(artifact.values_path)
        saved_paths.extend(list(plots.values()))

    logging.info("Generated %d plot files", len(saved_paths))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    raise SystemExit(main(output_root=args.output_root))
