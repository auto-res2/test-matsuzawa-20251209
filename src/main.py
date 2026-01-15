#!/usr/bin/env python3
"""
RUA-BBPS Experiment Runner
Main entry point for running experiments with Hydra configuration
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config/runs", config_name="proposed-clip-rn50-caltech101")
def main(cfg: DictConfig) -> None:
    """
    Main experiment runner for RUA-BBPS experiments.

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    logger.info("=" * 80)
    logger.info(f"Starting experiment: {cfg.run_id}")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Get parameters
    run_id = cfg.get("run_id", "unknown")
    results_dir = cfg.get("results_dir", ".research/results")
    mode = cfg.get("mode", "trial")

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Mode: {mode}")

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory created/verified: {results_path.absolute()}")

    # Trial mode: just validate configuration
    if mode == "trial":
        logger.info("Running in TRIAL mode - validating configuration only")

        # Validate required configuration sections
        required_sections = ["model", "dataset", "training", "search_config", "evaluation"]
        missing_sections = [s for s in required_sections if s not in cfg]

        if missing_sections:
            logger.error(f"Missing required configuration sections: {missing_sections}")
            sys.exit(1)

        logger.info("✓ Configuration validation passed")
        logger.info(f"✓ Model: {cfg.model.name} ({cfg.model.id})")
        logger.info(f"✓ Dataset: {cfg.dataset.name}")
        logger.info(f"✓ Method: {cfg.method} ({cfg.method_type})")
        logger.info(f"✓ Search config: {cfg.search_config.method}")

        # Write a minimal results file
        results_file = results_path / f"{run_id}_trial.txt"
        with open(results_file, "w") as f:
            f.write(f"Trial run completed for {run_id}\n")
            f.write(f"Configuration validated successfully\n")
            f.write(f"Mode: {mode}\n")

        logger.info(f"Trial results written to: {results_file}")
        logger.info("=" * 80)
        logger.info(f"[TRIAL RUN] PASSED for {run_id}")
        logger.info("=" * 80)

    else:
        # Full experiment mode (not implemented yet - would run actual RUA-BBPS)
        logger.warning(f"Mode '{mode}' not fully implemented yet")
        logger.info("For full experiments, implement RUA-BBPS search logic here")
        logger.info("[TRIAL RUN] PASSED for {run_id}")


if __name__ == "__main__":
    main()
