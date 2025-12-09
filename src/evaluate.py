# src/evaluate.py
import os
import sys
import json
import argparse
from pathlib import Path
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wandb.apis import PublicApi


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--run_ids", required=True, help='JSON string list of run ids')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)

    # Load wandb config file from config/config.yaml in repo root
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("config/config.yaml not found in repository root")
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    entity = cfg.get("wandb", {}).get("entity", "test")
    project = cfg.get("wandb", {}).get("project", "test")

    api = wandb.Api()

    aggregated = {
        "primary_metric": "held-out top-1 accuracy (averaged across dataset folds and restarts)",
        "metrics": {},
        "best_proposed": {"run_id": None, "value": None},
        "best_baseline": {"run_id": None, "value": None},
        "gap": None
    }

    per_run_results = {}

    for run_id in run_ids:
        print(f"Processing run_id={run_id}")
        run_dir = results_dir / run_id
        ensure_dir(run_dir)
        try:
            run = api.run(f"{entity}/{project}/{run_id}")
        except Exception as e:
            print(f"Failed to fetch run {run_id} from W&B: {e}")
            continue
        history = run.history()  # pandas DataFrame
        summary = run.summary._json_dict
        config = dict(run.config)

        # Save comprehensive metrics
        metrics_out = run_dir / "metrics.json"
        # Convert history to JSON serializable
        hist_json = history.fillna(0).to_dict(orient="list")
        out = {"history": hist_json, "summary": summary, "config": config}
        with open(metrics_out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"Saved run metrics to {metrics_out}")

        # Generate learning curve figure
        try:
            plt.figure(figsize=(6, 4))
            if "val_acc" in history.columns:
                plt.plot(history.index, history["val_acc"].values, label="val_acc")
            if "acc" in history.columns:
                plt.plot(history.index, history["acc"].values, label="train_acc")
            plt.xlabel("step")
            plt.ylabel("accuracy")
            plt.title(f"Learning curve {run_id}")
            plt.legend()
            fname = run_dir / f"{run_id}_learning_curve.pdf"
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"Saved figure: {fname}")
        except Exception as e:
            print(f"Failed to generate learning curve for {run_id}: {e}")

        # Confusion matrix not possible without per-sample preds; skip but generate placeholder
        # Aggregate primary metric from summary if available
        primary_val = summary.get("final_test_acc") or summary.get("best_val_acc") or None
        per_run_results[run_id] = {"primary": primary_val, "summary": summary}

    # Aggregated analysis
    # Collect primary metric across runs
    metrics = {}
    for run_id, data in per_run_results.items():
        for k, v in (data.get("summary") or {}).items():
            metrics.setdefault(k, {})[run_id] = v
    aggregated["metrics"] = metrics

    # Determine best proposed / best baseline
    prop_runs = [r for r in per_run_results.keys() if "proposed" in r or "proposed" in (per_run_results[r]["summary"].get("method", "") if per_run_results[r].get("summary") else "")] 
    base_runs = [r for r in per_run_results.keys() if ("comparative" in r or "comparative" in (per_run_results[r]["summary"].get("method", "") if per_run_results[r].get("summary") else "") or "Raw hill_climb" in (per_run_results[r]["summary"].get("method", "") if per_run_results[r].get("summary") else ""))]

    def get_primary(run_id):
        s = per_run_results[run_id]["summary"]
        for key in ["final_test_acc", "best_val_acc", "final_acc"]:
            if key in s:
                return float(s[key])
        return None

    best_prop = (None, -1.0)
    for r in prop_runs:
        v = get_primary(r)
        if v is not None and v > best_prop[1]:
            best_prop = (r, v)
    best_base = (None, -1.0)
    for r in base_runs:
        v = get_primary(r)
        if v is not None and v > best_base[1]:
            best_base = (r, v)

    aggregated["best_proposed"] = {"run_id": best_prop[0], "value": best_prop[1]}
    aggregated["best_baseline"] = {"run_id": best_base[0], "value": best_base[1]}

    # Gap calculation: percentage improvement over baseline
    if best_prop[1] is not None and best_base[1] and best_base[1] > 0:
        gap = (best_prop[1] - best_base[1]) / best_base[1] * 100.0
        aggregated["gap"] = float(gap)
    else:
        aggregated["gap"] = None

    # Save aggregated metrics
    comp_dir = results_dir / "comparison"
    ensure_dir(comp_dir)
    agg_out = comp_dir / "aggregated_metrics.json"
    with open(agg_out, "w") as fh:
        json.dump(aggregated, fh, indent=2, default=str)
    print(f"Saved aggregated metrics to {agg_out}")

    # Comparison figure: bar chart of primary metric across runs
    try:
        runs = []
        vals = []
        for r in per_run_results.keys():
            v = get_primary(r)
            if v is None:
                continue
            runs.append(r)
            vals.append(v)
        if runs:
            plt.figure(figsize=(max(6, len(runs) * 1.2), 4))
            sns.barplot(x=runs, y=vals)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('primary_metric')
            plt.title('Primary metric comparison')
            fname = comp_dir / "comparison_primary_metric_bar_chart.pdf"
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            print(f"Saved comparison figure: {fname}")
    except Exception as e:
        print(f"Failed to generate comparison figure: {e}")

    # Print generated file paths
    for run_id in run_ids:
        run_subdir = results_dir / run_id
        if run_subdir.exists():
            for p in run_subdir.iterdir():
                print(str(p))
    for p in comp_dir.iterdir():
        print(str(p))


if __name__ == '__main__':
    main()
