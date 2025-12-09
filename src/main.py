# src/main.py
import os
import sys
import subprocess
from pathlib import Path
from omegaconf import OmegaConf
import hydra

@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    # This main orchestrator launches a single run's train.py as a subprocess using Hydra overrides
    # Expected CLI: python -m src.main run={run_id} results_dir={path} mode=full
    run_id = cfg.get("run", cfg.run.run_id if hasattr(cfg, "run") else None)
    results_dir = getattr(cfg, "results_dir", "results")
    mode = getattr(cfg, "mode", None)
    if run_id is None or mode is None:
        print("Usage: python -m src.main run={run_id} results_dir={path} mode=<trial|full>")
        sys.exit(1)

    # Adjust wandb behavior according to mode
    if mode == "trial":
        # ensure trial settings
        overrides = [f"run.run_id={run_id}", f"results_dir={results_dir}", "mode=trial"]
    else:
        overrides = [f"run.run_id={run_id}", f"results_dir={results_dir}", "mode=full"]

    cmd = [sys.executable, "-u", "-m", "src.train"] + overrides
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Train subprocess failed with return code {proc.returncode}")

if __name__ == '__main__':
    main()
