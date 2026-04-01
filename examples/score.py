"""
load_and_score.py
-----------------
Score custom prompts with a pre-fitted LatentBiopsy reference profile.

Single prompt:
    python examples/load_and_score.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --biomarker biomarkers/qwen2.5-0.5b-instruct_layer20.pkl \
        --prompt "Write a step-by-step guide on how to bypass a firewall."

Batch file:
    python examples/load_and_score.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --biomarker biomarkers/qwen2.5-0.5b-instruct_layer20.pkl \
        --input-file user_logs.txt \
        --output-file anomaly_scores.csv
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
    p = argparse.ArgumentParser(description="Score custom prompts using a saved LatentBiopsy profile.")
    p.add_argument("--model", required=True, help="HuggingFace model ID.")
    p.add_argument("--biomarker", required=True, help="Path to the saved biomarker profile (.pkl).")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt", help="A single prompt string to score.")
    mode.add_argument("--input-file", help="Plain-text file with one prompt per line (batch mode).")
    p.add_argument("--output-file", default="anomaly_scores.csv", help="Destination CSV for batch scores.")
    return p

def main() -> None:
    args = _build_parser().parse_args()

    print(f"[score] Loading biomarker from '{args.biomarker}'...")
    try:
        biomarker = ThetaBiomarker.load(args.biomarker)
    except FileNotFoundError:
        sys.exit(f"[score] Error: biomarker file not found at '{args.biomarker}'.")

    stored = getattr(biomarker, "model_id", None)
    if stored is not None and stored != args.model:
        print(f"[score] Warning: biomarker fitted on '{stored}', but --model is '{args.model}'.")

    print(f"[score] Loading model '{args.model}'...")
    extractor = LatentExtractor(args.model)

    if args.prompt:
        print(f"[score] Prompt: '{args.prompt}'")
        act = extractor.get_last_token_activations(args.prompt)
        score = biomarker.score(act)
        print(f"[score] Anomaly score (-log p): {score:.4f}")
        return

    prompt_path = Path(args.input_file)
    if not prompt_path.exists():
        sys.exit(f"[score] Error: input file not found at '{args.input_file}'.")
    with prompt_path.open(encoding="utf-8") as fh:
        prompts =[line.strip() for line in fh if line.strip()]

    print(f"[score] Extracting activations for {len(prompts)} prompts...")
    acts = torch.stack([extractor.get_last_token_activations(p) for p in prompts])

    print("[score] Computing anomaly scores...")
    scores = biomarker.score_batch(acts)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("prompt,anomaly_score\n")
        for prompt, score in zip(prompts, scores):
            safe_prompt = prompt.replace('"', '""')
            fh.write(f'"{safe_prompt}",{score:.6f}\n')

    print(f"[score] Saved {len(prompts)} scores to '{output_path}'.")

if __name__ == "__main__":
    main()