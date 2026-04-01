import argparse
import sys
import torch
import json
import pandas as pd
from pathlib import Path
from latentbiopsy.extraction import LatentExtractor
from latentbiopsy.theta import ThetaBiomarker

def resolve_layer(layer_input: str, total_layers: int) -> int:
    """Handles 'last' or negative indexing like -1, -2."""
    if layer_input.lower() == "last":
        return total_layers - 1
    try:
        idx = int(layer_input)
        if idx < 0:
            return total_layers + idx
        return idx
    except ValueError:
        sys.exit(f"Error: Layer must be an integer or 'last', got {layer_input}")

def fit_main():
    p = argparse.ArgumentParser(description="LatentBiopsy: Fit a safety profile.")
    p.add_argument("--model", required=True)
    p.add_argument("--normative-file", required=True)
    p.add_argument("--layer", default="-1", help="Layer index (supports negative indexing, e.g., -1).")
    p.add_argument("--N", type=int, default=200, help="Number of prompts (default: 200).")
    p.add_argument("--output", required=True, help="Path to save .pkl file.")
    args = p.parse_args()

    # 1. Load data
    with open(args.normative_file, "r", encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()][:args.N]
    
    # 2. Extract
    extractor = LatentExtractor(args.model)
    layer_idx = resolve_layer(args.layer, extractor.num_layers)
    
    print(f"Extracting activations at resolved layer {layer_idx}...")
    acts = torch.stack([extractor.get_last_token_activations(p) for p in prompts])
    
    # 3. Fit & Save
    bm = ThetaBiomarker(layer_indices=[layer_idx])
    bm.fit(acts)
    bm.save(args.output, model_id=args.model, fit_n=len(prompts))
    print(f"Successfully saved biomarker to {args.output}")

def score_main():
    p = argparse.ArgumentParser(description="LatentBiopsy: Score prompts.")
    p.add_argument("--model", required=True)
    p.add_argument("--biomarker", required=True)
    
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Single string to score.")
    group.add_argument("--input-file", type=str, help="Path to .txt or .jsonl file.")
    
    p.add_argument("--output", type=str, help="Optional path to save results as CSV.")
    args = p.parse_args()

    # 1. Initialization
    print(f"Loading biomarker...")
    bm = ThetaBiomarker.load(args.biomarker)
    print(f"Loading model...")
    extractor = LatentExtractor(args.model)

    # 2. Parse Prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.input_file:
        input_path = Path(args.input_file)
        
        if not input_path.exists():
            sys.exit(f"Error: File '{args.input_file}' not found.")

        # --- Enhanced JSON/JSONL Handling ---
        if input_path.suffix.lower() in [".json", ".jsonl"]:
            with open(input_path, "r", encoding="utf-8") as f:
                if input_path.suffix.lower() == ".jsonl":
                    # Parse JSONL (line by line)
                    raw_data = [json.loads(line) for line in f if line.strip()]
                else:
                    # Parse standard JSON (whole file)
                    raw_data = json.load(f)
                    if isinstance(raw_data, dict):
                        raw_data = [raw_data] # Wrap single object in list

                # Extract prompts using common keys
                for entry in raw_data:
                    p_text = entry.get("prompt") or entry.get("text") or entry.get("instruction")
                    if p_text:
                        prompts.append(p_text)
        # --- End Enhanced Handling ---
        
        else:
            # Default to plain text (one prompt per line)
            with open(input_path, "r", encoding="utf-8") as f:
                prompts = [l.strip() for l in f if l.strip()]

    if not prompts:
        sys.exit("Error: No prompts found to score.")

    # 3. Extract and Score
    print(f"Scoring {len(prompts)} prompt(s)...")
    # Batch processing via stack
    acts = torch.stack([extractor.get_last_token_activations(p) for p in prompts])
    scores = bm.score_batch(acts)

    # 4. Handle Results
    results_list = []
    for p, s in zip(prompts, scores):
        score_val = round(float(s), 4)
        results_list.append({"prompt": p, "anomaly_score": score_val})
        print(f"{score_val:.4f} | {p}")

    # 5. Optional Export
    if args.output:
        df = pd.DataFrame(results_list)
        df.to_csv(args.output, index=False)
        print(f"\n✅ Results exported to {args.output}")

if __name__ == "__main__":
    score_main()