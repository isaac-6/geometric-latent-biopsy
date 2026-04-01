"""
fit_and_save.py
---------------
Fit a LatentBiopsy reference profile on a set of normative (safe) prompts
and save it to disk for later inference.

Usage:
    python examples/fit_and_save.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --normative-file data/raw/normative.txt \
        --layer 20 \
        --output biomarkers/qwen2.5-0.5b-instruct_layer20.pkl \
        --N 200
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from latentbiopsy.extraction import LatentExtractor
from latentbiopsy.theta import ThetaBiomarker

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit a LatentBiopsy reference profile on normative prompts.")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path.")
    p.add_argument("--normative-file", required=True, help="Plain-text file with one safe prompt per line.")
    p.add_argument("--layer", type=int, required=True, help="Residual-stream layer index to extract.")
    p.add_argument("--N", type=int, default=200, help="Number of normative prompts to use for fitting.")
    p.add_argument("--output", required=True, help="Destination path for the saved biomarker profile (.pkl).")
    return p

def main() -> None:
    args = _build_parser().parse_args()
    normative_path = Path(args.normative_file)
    if not normative_path.exists():
        sys.exit(f"[fit] Error: normative file not found at '{normative_path}'.")

    with normative_path.open(encoding="utf-8") as fh:
        all_prompts = [line.strip() for line in fh if line.strip()]

    n_available = len(all_prompts)
    if n_available == 0:
        sys.exit(f"[fit] Error: '{normative_path}' is empty.")

    n_fit = min(args.N, n_available)
    print(f"[fit] Using {n_fit} of {n_available} available normative prompts.")
    prompts = all_prompts[:n_fit]

    print(f"[fit] Loading model '{args.model}'...")
    extractor = LatentExtractor(args.model)

    print(f"[fit] Extracting activations at layer {args.layer} for {n_fit} prompts...")
    acts = torch.stack([extractor.get_last_token_activations(p) for p in prompts])

    print("[fit] Fitting ThetaBiomarker...")
    biomarker = ThetaBiomarker(layer_indices=[args.layer])
    biomarker.fit(acts)

    output_path = Path(args.output)
    biomarker.save(output_path, model_id=args.model, fit_n=n_fit)
    print(f"[fit] Done. Reference profile saved to '{output_path}'.")
    print(f"[fit] To score prompts: python examples/load_and_score.py --model {args.model} --biomarker {output_path} ...")

if __name__ == "__main__":
    main()