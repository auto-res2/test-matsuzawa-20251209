# src/train.py
import os
import sys
import json
import time
import random
import math
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Caltech101, OxfordIIITPet

import wandb
import numpy as np
from omegaconf import OmegaConf
import hydra

# Ensure transformers cache is kept in .cache/
os.environ.setdefault("TRANSFORMERS_CACHE", ".cache/")

from transformers import CLIPProcessor, CLIPModel

# Utilities (MinHash, tokenization, bootstrap) based on the paper-method snippet
import re
import hashlib
from collections import defaultdict


def tokenize_prompt(prompt: str):
    return re.findall(r"\w+", prompt.lower())


def ngrams(tokens, n=3):
    return [" ".join(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def minhash_signature(tokens, n=3, k_sig=64):
    shingles = ngrams(tokens, n=n)
    if not shingles:
        return tuple([2**64 - 1] * k_sig)
    sig = []
    for i in range(k_sig):
        minh = 2**64 - 1
        for s in shingles:
            h = int(hashlib.sha256((s + "|" + str(i)).encode()).hexdigest()[:16], 16)
            if h < minh:
                minh = h
        sig.append(minh)
    return tuple(sig)


def minhash_jaccard(sig_a, sig_b):
    assert len(sig_a) == len(sig_b)
    match = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return match / float(len(sig_a))


def length_norm(prompt, max_len=40):
    toks = tokenize_prompt(prompt)
    return min(len(toks) / float(max_len), 1.0)


# Simple CLIP-based evaluator wrapper (uses transformers' CLIPModel)
class CLIPEvaluator:
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model()

    def _init_model(self):
        # Load model + processor (tokenizer+image preprocess)
        try:
            self.model = CLIPModel.from_pretrained(self.model_id, cache_dir=".cache/")
            self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=".cache/")
        except Exception as e:
            print("Failed to load CLIPModel from transformers:", e, file=sys.stderr)
            raise
        self.model.to(self.device)
        self.model.eval()
        # Defensive check: tokenizer pad token
        try:
            tok = self.processor.tokenizer
            if getattr(tok, "pad_token_id", None) is None:
                # set to eos_token if missing
                if getattr(tok, "eos_token_id", None) is not None:
                    tok.pad_token_id = tok.eos_token_id
                else:
                    tok.add_special_tokens({"pad_token": "<|pad|>"})
        except Exception:
            pass

    @torch.no_grad()
    def encode_images(self, pil_images: List[Any]):
        # pil_images: list of PIL images
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_embeds = self.model.get_image_features(**inputs)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds.cpu()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_embeds = self.model.get_text_features(**inputs)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds.cpu()


# Simple local/mock proposer if OpenAI is not used
class MockProposer:
    def __init__(self, seed: int = 0):
        random.seed(seed)
        self.templates = [
            "A photo of a {label}.",
            "An image of the {label} species.",
            "This is a picture of a {label}.",
            "A close-up photo of a {label} in natural light.",
            "A low angle shot of a {label}.",
            "A {label} on a plain background.",
            "A vintage photograph of a {label}.",
        ]

    def propose(self, current_pool: List[str], topk: int, n_proposals: int) -> List[str]:
        # Generate proposals by mutating top prompts and combining template variations
        proposals = []
        base_phrases = ["A photo of", "An image of", "A picture of", "A close-up of", "A cropped image of"]
        words = ["vivid", "clear", "in nature", "on a plain background", "studio shot", "closeup"]
        for _ in range(n_proposals):
            template = random.choice(self.templates)
            # pick a class-like placeholder to format later; here use '{label}' placeholder left intact
            prompt = template
            # random suffix/prefix
            if random.random() < 0.4:
                prompt = random.choice(base_phrases) + " {label}."
            if random.random() < 0.5:
                prompt = prompt.replace(".", ", " + random.choice(words) + ".")
            # Mutate by adding an adjective
            if random.random() < 0.3:
                prompt = prompt.replace("{label}", "{label} {adj}")
                prompt = prompt.replace("{adj}", random.choice(["close-up", "cute", "adult", "young"]))
            proposals.append(prompt)
        # ensure uniqueness
        return list(dict.fromkeys(proposals))


# Simple helper: evaluate a single prompt on a labeled dataset using CLIP similarity
# prompt is a template string containing "{label}" placeholder

def evaluate_prompt_on_indices(prompt_template: str, class_names: List[str], images, labels, indices: List[int], clip_eval: CLIPEvaluator, batch_size=64):
    # Build texts by filling {label}
    texts = [prompt_template.format(label=cn) for cn in class_names]
    # encode texts once
    text_embeds = clip_eval.encode_texts(texts)  # (num_classes, D)
    correct = 0
    total = 0
    # iterate images at indices
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        imgs = [images[idx] for idx in batch_idx]
        image_embeds = clip_eval.encode_images(imgs)  # (B, D)
        # similarity
        sims = image_embeds @ text_embeds.t()  # (B, C)
        preds = sims.argmax(dim=1).numpy().tolist()
        for p, idx in zip(preds, batch_idx):
            if p == labels[idx]:
                correct += 1
        total += len(batch_idx)
    return float(correct) / float(max(1, total))


# Bootstrap-based uncertainty estimator
def bootstrap_std_estimate(fn_eval_on_indices, prompt_template, class_names, indices, B=20, sample_frac=0.7):
    if len(indices) == 0:
        return 0.0
    vals = []
    for _ in range(B):
        k = max(1, int(len(indices) * sample_frac))
        subset = [random.choice(indices) for _ in range(k)]
        vals.append(fn_eval_on_indices(prompt_template, class_names, subset))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)


# Adaptive multiplicative weight updater (discrete grid)
class AdaptiveCoeffUpdater:
    def __init__(self, grid: List[tuple], eta: float = 0.2):
        self.grid = grid
        self.weights = [1.0 for _ in grid]
        self.eta = eta

    def sample(self):
        total = sum(self.weights)
        r = random.random() * total
        cum = 0.0
        for i, w in enumerate(self.weights):
            cum += w
            if r <= cum:
                return i, self.grid[i]
        return len(self.grid) - 1, self.grid[-1]

    def update(self, idx: int, reward: float):
        # reward in [0,1]
        self.weights[idx] *= math.exp(self.eta * reward)


# Main training routine
@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    # Hydra will change cwd; ensure outputs use user-provided results_dir if given
    # Hydra passes overrides; ensure we can access results_dir override or default to ./outputs
    results_dir = getattr(cfg, "results_dir", None) or os.environ.get("RESULTS_DIR") or "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Mode adjustments
    mode = getattr(cfg, "mode", "full")
    if mode == "trial":
        # lightweight: disable wandb, tiny epochs/trials
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        # reduce expensive budgets if present
        if hasattr(cfg.training, "iterations_per_restart"):
            cfg.training.iterations_per_restart = 1
        if hasattr(cfg.training, "proposals_per_iteration"):
            cfg.training.proposals_per_iteration = min(2, int(cfg.training.proposals_per_iteration))
    elif mode == "full":
        cfg.wandb.mode = cfg.wandb.get("mode", "online")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Initialize wandb unless disabled
    wandb_run = None
    if cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(entity=cfg.wandb.entity,
                               project=cfg.wandb.project,
                               id=cfg.run.run_id,
                               config=OmegaConf.to_container(cfg, resolve=True),
                               resume="allow")
    else:
        print("WandB disabled for trial mode")

    print(f"Starting run {cfg.run.run_id} mode={mode} results_dir={str(results_dir)}")

    # Set random seeds
    seed = int(getattr(cfg.run, "seed", 0) or 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize CLIP evaluator
    clip_model_id = cfg.model.get("id", "openai/clip-vit-base-patch32")
    clip_eval = CLIPEvaluator(model_id=clip_model_id)

    # Defensive assertions after init
    # Assert model outputs dims
    try:
        # make a small forward pass for shapes
        txts = ["a photo of a cat", "a photo of a dog"]
        te = clip_eval.encode_texts(txts)
        ie = clip_eval.encode_images([clip_eval.processor.feature_extractor.image_mean] * 1)
        assert te.ndim == 2
    except Exception:
        # Not critical to stop, but must assert
        print("Post-init assertion failed: CLIP encoding smoke test failed", file=sys.stderr)
        raise

    # Load dataset via preprocess utilities
    dataset_cfg = cfg.dataset
    dataset_name = dataset_cfg.get("id", dataset_cfg.get("name", "caltech101"))
    dataset_root = ".cache/"
    transform = transforms.Compose([
        transforms.Resize(cfg.dataset.preprocessing.image_size),
        transforms.CenterCrop(cfg.dataset.preprocessing.image_size) if cfg.dataset.preprocessing.center_crop else transforms.CenterCrop(cfg.dataset.preprocessing.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_eval.processor.image_mean, std=clip_eval.processor.image_std) if getattr(clip_eval.processor, "image_mean", None) is not None else transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if dataset_name.lower().startswith("caltech"):
        ds_full = Caltech101(root=dataset_root, download=True, transform=transform)
        # Caltech101 has .targets and .categories
        class_to_idx = {c: i for i, c in enumerate(ds_full.categories)}
        all_classes = ds_full.categories
    elif "pet" in dataset_name.lower() or "oxford" in dataset_name.lower():
        ds_full = OxfordIIITPet(root=dataset_root, download=True, transform=transform, target_types="category")
        # torchvision's OxfordIIITPet returns (img, target) target is index
        all_classes = sorted(set([ds_full._labels[idx] for idx in range(len(ds_full))])) if hasattr(ds_full, "_labels") else [str(i) for i in range(100)]
        # fallback: create pseudo-class names
        if len(all_classes) <= 1:
            all_classes = [f"class_{i}" for i in range(30)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Build simple lists of images and labels for our evaluator
    images = []
    labels = []
    class_names = []
    # For Caltech101, ds_full returns (PILImage, label_str) sometimes
    # We'll iterate and collect up to max_sampled_classes per config
    max_classes = int(cfg.dataset.get("max_sampled_classes", 30))
    # Build mapping of class -> indices
    class_to_indices = defaultdict(list)
    for idx in range(len(ds_full)):
        item = ds_full[idx]
        if isinstance(item, tuple) and len(item) >= 2:
            img, target = item[0], item[1]
        else:
            continue
        # target can be int or str
        if isinstance(target, int):
            lbl = str(target)
        else:
            lbl = str(target)
        class_to_indices[lbl].append(idx)
        images.append(img)
        labels.append(int(target) if isinstance(target, int) else 0)
    # Build class_names limited to max_classes
    available_classes = list(class_to_indices.keys())
    if len(available_classes) == 0:
        # fallback: use 10 dummy classes
        available_classes = [f"class_{i}" for i in range(min(30, len(images) or 30))]
    sampled_classes = available_classes[:max_classes]
    class_names = sampled_classes

    # Build indices mapping for sampled classes
    indices_per_class = {c: class_to_indices[c][:200] for c in sampled_classes if c in class_to_indices}
    # Flatten and build labels mapping by index for evaluator
    # For simplicity, we will create labels by mapping class to local index
    class_to_local = {c: i for i, c in enumerate(sampled_classes)}
    images_list = []
    labels_list = []
    index_map = []
    for c in sampled_classes:
        idxs = indices_per_class.get(c, [])
        for original_idx in idxs:
            item = ds_full[original_idx]
            img, _ = item[0], item[1]
            images_list.append(img)
            labels_list.append(class_to_local[c])
            index_map.append(original_idx)
    # If empty, fall back to first N items
    if len(images_list) == 0:
        for i in range(min(200, len(ds_full))):
            item = ds_full[i]
            images_list.append(item[0])
            labels_list.append(0)
    images = images_list
    labels = labels_list

    # Build train/val/test split indices randomly per fold (runner-scale small)
    n_folds = int(cfg.dataset.splits.get("n_folds", 3))
    shots = cfg.dataset.splits.get("shots", [1])

    # Simple split: shuffle indices then slice
    indices_all = list(range(len(images)))
    random.shuffle(indices_all)
    fold_splits = []
    fold_size = max(1, len(indices_all) // n_folds)
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size
        fold_indices = indices_all[start:end]
        # within each fold, create few-shot train and held-out test
        # few-shot train: sample k*#classes images randomly from fold
        fold_splits.append(fold_indices)

    # Prepare proposer
    proposer_cfg = cfg.components.get("proposer", {}) if hasattr(cfg, "components") else {}
    proposer_mode = proposer_cfg.get("mode", "production")
    prefer_local = cfg.search_config.get("cost_saving_defaults", {}).get("prefer_local_mock_proposer", False)
    use_mock = (proposer_mode != "production") or prefer_local or (os.environ.get("OPENAI_API_KEY") is None)
    if use_mock:
        proposer = MockProposer(seed=seed)
    else:
        # If OpenAI available, define a wrapper that calls their ChatCompletion API
        try:
            import openai
            openai.api_key = os.environ.get("OPENAI_API_KEY")

            class OpenAIProposer:
                def __init__(self, model_name="gpt-3.5-turbo", temperature=0.8):
                    self.model = model_name
                    self.temperature = temperature

                def propose(self, current_pool, topk, n_proposals):
                    prompt = "Generate %d prompt templates for image classification. Current top prompts: %s" % (n_proposals, json.dumps(current_pool[:topk]))
                    resp = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": prompt}], temperature=self.temperature)
                    txt = resp["choices"][0]["message"]["content"]
                    # split by lines
                    proposals = [l.strip() for l in txt.splitlines() if len(l.strip()) > 0]
                    if not proposals:
                        # fallback to mock
                        return MockProposer().propose(current_pool, topk, n_proposals)
                    return proposals

            proposer = OpenAIProposer(model_name=proposer_cfg.get("id", "gpt-3.5-turbo"))
        except Exception:
            proposer = MockProposer(seed=seed)

    # Build coefficient grid for adaptive updater (small discrete grid)
    base_lambda = cfg.search_config.length_normalization.get("default_lambda_len", 0.05)
    base_mu = cfg.search_config.semantic_penalty.get("default_mu_sim", 0.3)
    base_gamma = cfg.search_config.uncertainty_penalty.get("default_gamma_unc", 0.5)
    grid = []
    for l in [max(0.0, base_lambda * f) for f in [0.5, 1.0, 2.0]]:
        for m in [max(0.0, base_mu * f) for f in [0.5, 1.0, 2.0]]:
            for g in [max(0.0, base_gamma * f) for f in [0.5, 1.0, 2.0]]:
                grid.append((l, m, g))
    updater = AdaptiveCoeffUpdater(grid=grid, eta=cfg.search_config.adaptive_updater.get("eta", 0.2))

    # Main hill-climb loop (runner-scale)
    nrestart = int(cfg.training.get("nrestart", 5))
    iterations_per_restart = int(cfg.training.get("iterations_per_restart", 5))
    proposals_per_iteration = int(cfg.training.get("proposals_per_iteration", 20))
    top_k_pool = int(cfg.training.get("top_k_pool", 10))
    validation_budget = int(cfg.training.get("validation_budget_per_restart", 3))

    # Bookkeeping
    run_metrics = {
        "run_id": cfg.run.run_id,
        "per_restart": []
    }

    total_llm_calls = 0

    for restart in range(nrestart):
        # For reproducibility vary seed
        random.seed(seed + restart)
        # Initialize prompt pool with simple templates
        pool = ["A photo of a {label}.", "An image of the {label} species."]
        pool = list(dict.fromkeys(pool))
        scores_cache = {}
        sig_cache = {}

        # sample fold and shot
        fold_idx = restart % n_folds
        fold_indices = fold_splits[fold_idx]
        k_shots = shots[0] if isinstance(shots, list) else shots
        # Build few-shot train indices: sample k_shots per class
        # Map class -> indices in fold
        fold_class_indices = defaultdict(list)
        for idx in fold_indices:
            lbl = labels[idx]
            fold_class_indices[lbl].append(idx)
        train_indices = []
        val_indices = []
        test_indices = []
        for lbl, idxs_for_lbl in fold_class_indices.items():
            random.shuffle(idxs_for_lbl)
            k = min(len(idxs_for_lbl), k_shots)
            train_indices.extend(idxs_for_lbl[:k])
            if len(idxs_for_lbl) > k:
                val_indices.extend(idxs_for_lbl[k:k + 1])
                test_indices.extend(idxs_for_lbl[k + 1:k + 3])
        if len(val_indices) == 0:
            # fallback: a few random indices
            val_indices = random.sample(fold_indices, min(5, len(fold_indices)))
        if len(test_indices) == 0:
            test_indices = [i for i in fold_indices if i not in train_indices][:min(30, len(fold_indices))]

        # Adaptive coefficients sampling initially
        coeff_idx, (lambda_len, mu_sim, gamma_unc) = updater.sample()

        restart_record = {"restart": restart, "best_prompt": None, "best_val_acc": 0.0, "history": []}

        for it in range(iterations_per_restart):
            # Sample proposals from proposer
            proposals = proposer.propose(pool, topk=top_k_pool, n_proposals=proposals_per_iteration)
            total_llm_calls += 1

            # Evaluate proposals: compute acc_train, len_norm, minhash sim vs top-k, bootstrap unc
            topk_sigs = [sig_cache.get(t) or minhash_signature(tokenize_prompt(t), n=cfg.search_config.minhash.get("ngram_n", 3), k_sig=cfg.search_config.minhash.get("k_sig", 64)) for t in pool[:top_k_pool]]

            for p in proposals:
                # Evaluate raw train acc if not cached
                if p not in scores_cache:
                    acc = evaluate_prompt_on_indices(p, class_names, images, labels, train_indices, clip_eval)
                    scores_cache[p] = {"acc": acc}
                else:
                    acc = scores_cache[p]["acc"]
                toks = tokenize_prompt(p)
                ln = length_norm(p, max_len=cfg.search_config.length_normalization.get("max_len_tokens", 40))
                sig = minhash_signature(toks, n=cfg.search_config.minhash.get("ngram_n", 3), k_sig=cfg.search_config.minhash.get("k_sig", 64))
                # semantic penalty: max similarity to topk
                max_sim = 0.0
                for s in topk_sigs:
                    sim = minhash_jaccard(sig, s)
                    if sim > max_sim:
                        max_sim = sim
                # uncertainty
                unc = bootstrap_std_estimate(lambda tmpl, cls, inds: evaluate_prompt_on_indices(tmpl, cls, images, labels, inds, clip_eval), p, class_names, train_indices, B=cfg.search_config.bootstrap.get("B", 20), sample_frac=cfg.search_config.bootstrap.get("sample_frac", 0.7))
                adj = acc - lambda_len * ln - mu_sim * max_sim - gamma_unc * unc
                scores_cache[p].update({"len": len(toks), "max_sim": max_sim, "unc": unc, "adj": adj, "sig": sig})
                sig_cache[p] = sig

                # Log per-proposal metrics
                logd = {f"iter": it, "restart": restart, "proposal": p, "acc": acc, "adj": adj, "len": len(toks), "max_sim": max_sim, "unc": unc}
                if wandb_run:
                    wandb.log({**logd, "total_llm_calls": total_llm_calls})

            # Combine pool and proposals, sort by adjusted score
            combined = list(dict.fromkeys(pool + proposals))
            combined_sorted = sorted(combined, key=lambda t: scores_cache.get(t, {}).get("adj", scores_cache.get(t, {}).get("acc", 0.0)), reverse=True)
            pool = combined_sorted[:cfg.search_config.pool_and_selection.get("top_k_for_similarity", top_k_pool * 2)]

            # Evaluate top-1 on held-out val occasionally per adaptive updater budget
            if (it % cfg.search_config.adaptive_updater.get("update_period_iterations", 3) == 0) and (validation_budget > 0):
                # Evaluate top candidate on val set
                top_candidate = pool[0]
                val_acc = evaluate_prompt_on_indices(top_candidate, class_names, images, labels, val_indices, clip_eval)
                # Reward scaled to [0,1]
                reward = float(val_acc)
                updater.update(coeff_idx, reward)
                # Resample coefficients
                coeff_idx, (lambda_len, mu_sim, gamma_unc) = updater.sample()
                # Record
                restart_record["history"].append({"iter": it, "top_candidate": top_candidate, "val_acc": val_acc, "lambda_len": lambda_len, "mu_sim": mu_sim, "gamma_unc": gamma_unc})
                if val_acc > restart_record["best_val_acc"]:
                    restart_record["best_val_acc"] = val_acc
                    restart_record["best_prompt"] = top_candidate
                if wandb_run:
                    wandb.log({"val_acc": val_acc, "lambda_len": lambda_len, "mu_sim": mu_sim, "gamma_unc": gamma_unc, "restart": restart, "iter": it})

        # End iterations per restart
        restart_record["total_llm_calls"] = total_llm_calls
        run_metrics["per_restart"].append(restart_record)

    # After all restarts: evaluate best found prompt on held-out test indices aggregated
    best_overall = None
    best_acc = -1.0
    for r in run_metrics["per_restart"]:
        if r["best_val_acc"] > best_acc:
            best_acc = r["best_val_acc"]
            best_overall = r["best_prompt"]
    final_test_acc = 0.0
    if best_overall:
        all_test_indices = []
        for fidx in range(n_folds):
            all_test_indices.extend(fold_splits[fidx])
        final_test_acc = evaluate_prompt_on_indices(best_overall, class_names, images, labels, all_test_indices, clip_eval)

    # Final logging and summary
    summary = {
        "best_prompt": best_overall,
        "best_val_acc": float(best_acc),
        "final_test_acc": float(final_test_acc),
        "total_llm_calls": int(total_llm_calls)
    }
    if wandb_run:
        wandb.summary.update(summary)
        print("WandB run url:", wandb.run.get_url())
    else:
        print("WandB disabled; run summary:\n", json.dumps(summary, indent=2))

    # Save metadata to results_dir
    run_dir = results_dir / cfg.run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_summary.json", "w") as fh:
        json.dump({**summary, "per_restart": run_metrics["per_restart"]}, fh, indent=2)

    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
