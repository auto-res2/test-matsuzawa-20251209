import argparse
import argparse
from pathlib import Path

import yaml


def load_yaml_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_configs(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_run_config(run_id: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    base_cfg = load_yaml_config(root / "config" / "config.yaml")
    run_cfg_path = root / "config" / "runs" / f"{run_id}.yaml"
    run_cfg = load_yaml_config(run_cfg_path)

    if run_cfg.get("run_id") != run_id:
        raise ValueError(
            f"Run config mismatch: expected run_id '{run_id}', got '{run_cfg.get('run_id')}'"
        )

    return merge_configs(base_cfg, run_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="RUA-BBPS experiment launcher")
    parser.add_argument(
        "--run-id",
        default="proposed-clip-rn50-caltech101",
        help="Name of the run YAML file (without extension) under config/runs",
    )
    args = parser.parse_args()

    cfg = resolve_run_config(args.run_id)

    print("=== RUA-BBPS Experiment Configuration ===")
    print(f"Run ID: {cfg['run']['run_id']}")
    print("Resolved configuration:")
    print(yaml.dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()
