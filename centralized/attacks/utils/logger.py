"""
Logging utility for all attack experiments.
Wraps W&B with a clean interface + always writes to local file
so experiments are reproducible even without W&B.

Usage:
    logger = Logger(experiment_name="label_flip", use_wandb=True)
    logger.log({"clean_accuracy": 0.94, "poisoned_accuracy": 0.61})
    logger.log_summary({"attack": "label_flip", "victim_label": 0, ...})
    logger.finish()
"""

import json
import time
from pathlib import Path
from datetime import datetime


class Logger:
    def __init__(
        self,
        experiment_name: str,
        config: dict = None,
        use_wandb: bool = False,
        log_dir: str = "./results/logs",
        project: str = "fl-attacks-lab",
    ):
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self.history = []
        self.start_time = time.time()

        # local log file — always written regardless of W&B
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{ts}.jsonl"

        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project=project,
                    name=f"{experiment_name}_{ts}",
                    config=config,
                )
                self._wandb = wandb
                print(f"[Logger] W&B run: {wandb.run.url}")
            except ImportError:
                print("[Logger] wandb not installed — logging locally only")
                self.use_wandb = False
            except Exception as e:
                print(f"[Logger] W&B init failed ({e}) — logging locally only")
                self.use_wandb = False
        else:
            self._wandb = None

        print(f"[Logger] Experiment: {experiment_name}")
        print(f"[Logger] Log file:   {self.log_file}")

    def log(self, metrics: dict, step: int = None):
        """Log a dict of metrics for one step/round."""
        entry = {"step": step, "time_elapsed": time.time() - self.start_time, **metrics}
        self.history.append(entry)

        # write to local file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # W&B
        if self.use_wandb and self._wandb:
            self._wandb.log(metrics, step=step)

    def log_summary(self, summary: dict):
        """Log final summary metrics (end of experiment)."""
        summary["experiment"] = self.experiment_name
        summary["total_time_s"] = round(time.time() - self.start_time, 2)

        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        if self.use_wandb and self._wandb:
            self._wandb.summary.update(summary)

        print(f"[Logger] Summary saved → {summary_file}")

    def finish(self):
        if self.use_wandb and self._wandb:
            self._wandb.finish()
        print(f"[Logger] Done. Total time: {time.time() - self.start_time:.1f}s")

    def get_history(self):
        return self.history
