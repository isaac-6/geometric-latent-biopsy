"""
run_model.py
------------
Single entry point for the full LatentBiopsy analysis pipeline.

For any given model, this script:
  1. Downloads datasets if absent                  (download_datasets.py)
  2. Auto-tunes the normative and harmful fit-N    (stability_analysis.py)
     by finding the smallest N where AUROC change
     per doubling falls below a tolerance threshold.
  3. Runs the full evaluation                      (evaluate_biomarker.py)
  4. Generates the theta-phi projection plots      (plot_theta_phi_full.py)
  5. Saves all outputs under results/<model_slug>/ with a manifest file
     recording the exact command and resolved parameters used.

This design means running a new model is a single command:

    python scripts/run_model.py --model Qwen/Qwen2.5-1.5B-Instruct

All outputs land in a self-contained directory.  Experiments are reproducible
from the manifest alone.

Usage
-----
    python scripts/run_model.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --normative-n  500 \\
        --harmful-n    520 \\
        --benign-agg-n 250 \\
        --seed 42 \\
        --stability-layers 0 6 12 19 22 \\
        --eval-layers all \\
        --strategy both \\
        --auroc-plateau-tol 0.01 \\
        --min-fit-n 50 \\
        --max-fit-n 400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Model slug: safe directory name from HuggingFace model ID
# ---------------------------------------------------------------------------

def model_slug(model_id: str) -> str:
    """Convert 'Qwen/Qwen2.5-0.5B-Instruct' → 'Qwen__Qwen2.5-0.5B-Instruct'."""
    return re.sub(r"[/\\]", "__", model_id)


# ---------------------------------------------------------------------------
# Auto-tuning: find the smallest N where AUROC plateaus
# ---------------------------------------------------------------------------

def auto_tune_fit_n(
    norm_acts:   torch.Tensor,   # (N_total, L, D)
    harm_acts:   torch.Tensor,   # (N_h, L, D)
    benign_acts: torch.Tensor,   # (N_b, L, D)
    target_layer: int,
    strategy: str,
    min_fit_n: int,
    max_fit_n: int,
    plateau_tol: float,
    seed: int,
) -> int:
    """
    Find the smallest fit-N where AUROC is within `plateau_tol` of the value
    at max_fit_n.  Uses a log-spaced grid and the harmful-vs-normative AUROC
    (normative_ref) or harmful-vs-normative AUROC post-orientation (harmful_ref).

    Algorithm
    ---------
    1. Fit at max_fit_n → AUROC_max (reference value).
    2. Walk the grid from small to large; stop at the first N where
       |AUROC(N) - AUROC_max| < plateau_tol.
    3. Return that N (or min_fit_n if none found).

    This is the quantitative operationalisation of the visual plateau judgment.
    """
    # Import here to avoid circular deps when run standalone
    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from theta import ThetaBiomarker   # type: ignore[import-untyped]
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)

    def _auroc(neg: np.ndarray, pos: np.ndarray) -> float:
        y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
        s = np.concatenate([neg, pos])
        if np.isnan(s).any() or len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, s))

    # Fixed eval sets (20% of each, shuffled)
    def _fixed_eval(acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N   = acts.shape[0]
        idx = rng.permutation(N)
        n_e = max(10, int(0.20 * N))
        return acts[idx[:N - n_e]], acts[idx[N - n_e:]]   # fit_pool, eval

    norm_fit_pool, norm_eval = _fixed_eval(norm_acts)
    harm_fit_pool, harm_eval = _fixed_eval(harm_acts)

    # Grid: log-spaced from min_fit_n to max_fit_n
    sizes = np.unique(np.round(
        np.geomspace(min_fit_n, min(max_fit_n, norm_fit_pool.shape[0]), num=20)
    ).astype(int)).tolist()
    if sizes[-1] < max_fit_n:
        sizes.append(min(max_fit_n, norm_fit_pool.shape[0]))

    def _score_at_n(n: int) -> float:
        if strategy == "normative_ref":
            fit_acts = norm_fit_pool[:n]
        else:
            n_h = min(n, harm_fit_pool.shape[0])
            fit_acts = harm_fit_pool[:n_h]

        bm = ThetaBiomarker(n_directions=1, n_gmm_components=1,
                            layer_indices=[target_layer])
        bm.fit(fit_acts)

        if strategy == "normative_ref":
            s_neg = bm.score_batch(norm_eval)
            s_pos = bm.score_batch(harm_eval)
        else:
            s_neg = -bm.score_batch(norm_eval)
            s_pos = -bm.score_batch(harm_eval)
            if float(np.mean(s_pos)) < float(np.mean(s_neg)):
                s_neg, s_pos = -s_neg, -s_pos

        return _auroc(s_neg, s_pos)

    auroc_max = _score_at_n(sizes[-1])
    print(f"    AUROC at max_fit_n={sizes[-1]}: {auroc_max:.4f}")

    plateau_n = sizes[-1]  # default: use maximum
    for n in sizes[:-1]:
        auroc_n = _score_at_n(n)
        gap = abs(auroc_n - auroc_max)
        print(f"    N={n:4d}  AUROC={auroc_n:.4f}  gap={gap:.4f}"
              f"  {'<= tol ✓' if gap <= plateau_tol else ''}")
        if gap <= plateau_tol:
            plateau_n = n
            break  # take the first (smallest) N within tolerance

    print(f"  => Plateau fit-N for {strategy}: {plateau_n}")
    return plateau_n


# ---------------------------------------------------------------------------
# Subprocess runner with logging
# ---------------------------------------------------------------------------

def run_step(cmd: list[str], log_dir: Path, step_name: str) -> None:
    """Run a subprocess, tee output to a log file, raise on failure."""
    log_path = log_dir / f"{step_name}.log"
    print(f"\n{'='*60}")
    print(f"  Step: {step_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")

    with open(log_path, "w") as log_fh:
        log_fh.write(f"# Command: {' '.join(cmd)}\n")
        log_fh.write(f"# Started: {datetime.now().isoformat()}\n\n")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"Step '{step_name}' failed (exit {proc.returncode}). "
                           f"See {log_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    slug  = model_slug(args.model)
    root  = Path(f"results/{slug}")
    root.mkdir(parents=True, exist_ok=True)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"\nLatentBiopsy pipeline — model: {args.model}")
    print(f"Output root: {root}")

    # ---- 1. Download datasets ----
    if not (Path("data/raw/normative.txt").exists() and
            Path("data/raw/harmful.txt").exists() and
            Path("data/raw/benign_aggressive.txt").exists()):
        run_step([
            sys.executable, "scripts/download_datasets.py",
            "--normative-n", str(args.normative_n),
            "--seed", str(args.seed),
        ], log_dir, "01_download_datasets")
    else:
        print("\n[skip] datasets already present in data/raw/")

    # ---- 2. Extract activations for auto-tuning ----
    print("\n[Auto-tune] Loading model and extracting activations for fit-N selection...")
    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from extraction import LatentExtractor   # type: ignore[import-untyped]

    def load_prompts(path: str, n: int) -> list[str]:
        with open(path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        random.seed(args.seed)
        return random.sample(lines, min(n, len(lines)))

    extractor = LatentExtractor(args.model)
    tune_layer = args.stability_layers[len(args.stability_layers) // 2]  # middle layer

    def extract(prompts: list[str]) -> torch.Tensor:
        return torch.stack([extractor.get_last_token_activations(p)
                            for p in prompts])

    norm_prompts   = load_prompts("data/raw/normative.txt",  args.normative_n)
    harm_prompts   = load_prompts("data/raw/harmful.txt",    args.harmful_n)
    benign_prompts = load_prompts("data/raw/benign_aggressive.txt", args.benign_agg_n)

    print("  Extracting normative activations...")
    norm_acts   = extract(norm_prompts)
    print("  Extracting harmful activations...")
    harm_acts   = extract(harm_prompts)
    print("  Extracting benign activations...")
    benign_acts = extract(benign_prompts)

    # ---- 3. Auto-tune fit-N for each strategy ----
    norm_fit_n = args.normative_fit_n
    harm_fit_n = args.harmful_fit_n

    strategies = (["normative_ref", "harmful_ref"]
                  if args.strategy == "both" else [args.strategy])

    if args.auto_tune:
        print("\n[Auto-tune] Normative-ref...")
        norm_fit_n = auto_tune_fit_n(
            norm_acts, harm_acts, benign_acts,
            target_layer=tune_layer, strategy="normative_ref",
            min_fit_n=args.min_fit_n, max_fit_n=args.max_fit_n,
            plateau_tol=args.auroc_plateau_tol, seed=args.seed,
        )
        if "harmful_ref" in strategies:
            print("\n[Auto-tune] Harmful-ref...")
            harm_fit_n = auto_tune_fit_n(
                norm_acts, harm_acts, benign_acts,
                target_layer=tune_layer, strategy="harmful_ref",
                min_fit_n=args.min_fit_n, max_fit_n=args.max_fit_n,
                plateau_tol=args.auroc_plateau_tol, seed=args.seed,
            )
    else:
        print(f"\n[Auto-tune skipped] Using fixed fit-N: "
              f"norm={norm_fit_n}, harm={harm_fit_n}")

    del extractor  # free GPU memory before subprocess calls

    # ---- 4. Stability analysis ----
    layers_str = [str(l) for l in args.stability_layers]
    eval_dir    = root / "eval"
    figures_dir = root / "figures"
    eval_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    run_step([
        sys.executable, "scripts/stability_analysis.py",
        "--model", args.model,
        "--normative-n", str(args.normative_n),
        "--harmful-n",   str(args.harmful_n),
        "--benign-agg-n", str(args.benign_agg_n),
        "--layers", *layers_str,
        "--target-layer", str(args.stability_layers[-1]),
        "--output-dir", str(eval_dir),
        "--seed", str(args.seed),
    ], log_dir, "02_stability")

    # ---- 5. Evaluation ----
    run_step([
        sys.executable, "scripts/evaluate_biomarker.py",
        "--model", args.model,
        "--normative-n",   str(args.normative_n),
        "--harmful-n",     str(args.harmful_n),
        "--benign-agg-n",  str(args.benign_agg_n),
        "--normative-fit-n", str(norm_fit_n),
        "--harmful-fit-n",   str(harm_fit_n),
        "--strategy", args.strategy,
        "--output-dir", str(eval_dir),
        "--seed", str(args.seed),
    ], log_dir, "03_evaluate")

    # ---- 6. Theta-phi plots ----
    for plot_layer in args.plot_layers:
        run_step([
            sys.executable, "scripts/plot_theta_phi_full.py",
            "--model", args.model,
            "--normative-n",    str(args.normative_n),
            "--harmful-n",      str(args.harmful_n),
            "--benign-agg-n",   str(args.benign_agg_n),
            "--normative-fit-n", str(norm_fit_n),
            "--harmful-fit-n",   str(harm_fit_n),
            "--layer", str(plot_layer),
            "--figures-dir", str(figures_dir),
            "--strategy", args.strategy,
            "--seed", str(args.seed),
        ], log_dir, f"04_theta_phi_layer{plot_layer}")

    # ---- 7. Write manifest ----
    manifest = {
        "model":               args.model,
        "output_root":         str(root),
        "eval_dir":            str(eval_dir),
        "figures_dir":         str(figures_dir),
        "timestamp":           datetime.now().isoformat(),
        "normative_fit_n":     norm_fit_n,
        "harmful_fit_n":       harm_fit_n,
        "auto_tune":           args.auto_tune,
        "auroc_plateau_tol":   args.auroc_plateau_tol,
        "auto_tune_layer":     tune_layer,
        "normative_n":         args.normative_n,
        "harmful_n":           args.harmful_n,
        "benign_agg_n":        args.benign_agg_n,
        "strategy":            args.strategy,
        "plot_layers":         args.plot_layers,
        "seed":                args.seed,
        "command":             " ".join(sys.argv),
    }
    manifest_path = root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved → {manifest_path}")
    print(f"\nAll done. Results in {root}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full LatentBiopsy pipeline for one model."
    )
    # Model
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model ID.")
    # Data sizes
    p.add_argument("--normative-n",  type=int, default=500)
    p.add_argument("--harmful-n",    type=int, default=520)
    p.add_argument("--benign-agg-n", type=int, default=250)
    # Fit-N (used if auto-tune is disabled)
    p.add_argument("--normative-fit-n", type=int, default=200,
                   help="Normative fit-N used when --no-auto-tune is set.")
    p.add_argument("--harmful-fit-n",   type=int, default=200,
                   help="Harmful fit-N used when --no-auto-tune is set.")
    # Auto-tuning
    p.add_argument("--auto-tune", action="store_true", default=True,
                   help="Automatically select fit-N from plateau analysis.")
    p.add_argument("--no-auto-tune", dest="auto_tune", action="store_false",
                   help="Skip auto-tuning and use --normative-fit-n / --harmful-fit-n.")
    p.add_argument("--auroc-plateau-tol", type=float, default=0.01,
                   help="AUROC gap threshold for plateau detection. "
                        "Plateau declared when |AUROC(N) - AUROC(max)| < tol. "
                        "Default: 0.01 (1 percentage point).")
    p.add_argument("--min-fit-n", type=int, default=20,
                   help="Minimum fit-N to consider during auto-tuning.")
    p.add_argument("--max-fit-n", type=int, default=400,
                   help="Maximum fit-N to consider during auto-tuning.")
    # Stability layers
    p.add_argument("--stability-layers", type=int, nargs="+",
                   default=[0, 6, 12, 19, 22],
                   help="Layers for stability analysis.")
    p.add_argument("--plot-layers",       type=int, nargs="+",
                   default=[5, 12, 19],
                   help="Layers for theta-phi projection plots. "
                        "Default: 5 (optimal detection), 12 (mid), 19 (best viz).")
    # Strategy
    p.add_argument("--strategy", default="normative_ref",
                   choices=["normative_ref", "harmful_ref", "both"])
    # Reproducibility
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()