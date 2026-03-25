"""
download_datasets.py
--------------------
Downloads and standardizes the three dataset splits used in the biopsy pipeline.

    - Normative / Safe : Alpaca-Cleaned (yahma/alpaca-cleaned) — instruction-only field
    - Harmful          : AdvBench harmful_behaviors (Zou et al., 2023)
    - Benign-Aggressive: XSTest safe subset (Röttger et al., 2023)

Each split is saved as a plain-text file with one prompt per line under:
    data/raw/<split>.txt

The script is idempotent: if the file already exists it is not re-downloaded.

Usage
-----
    python scripts/download_datasets.py [--normative-n 500] [--seed 42]

Dependencies
------------
    pip install datasets requests pandas
"""

import argparse
import os
import random
import sys
from typing import cast

NORMATIVE_OUT  = "data/raw/normative.txt"
HARMFUL_OUT    = "data/raw/harmful.txt"
BENIGN_AGG_OUT = "data/raw/benign_aggressive.txt"

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)

XSTEST_URL = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/"
    "main/xstest_prompts.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _write_lines(path: str, lines: list[str]):
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")
    print(f"  Saved {len(lines)} prompts → {path}")


def _already_exists(path: str) -> bool:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"  [skip] {path} already present ({n} prompts).")
        return True
    return False

def _is_sufficient(path: str, expected_n: int) -> bool:
    """Checks if the file exists and contains at least expected_n lines."""
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    if n < expected_n:
        print(f"  [update] {path} has {n} prompts; requested {expected_n}. Re-generating...")
        return False
    print(f"  [skip] {path} already sufficient ({n} prompts).")
    return True


# ---------------------------------------------------------------------------
# AdvBench — harmful behaviors
# ---------------------------------------------------------------------------

def download_advbench():
    print("\n[1/3] AdvBench — harmful_behaviors")
    if _already_exists(HARMFUL_OUT):
        return

    try:
        import requests
        import pandas as pd
    except ImportError:
        sys.exit("Install missing deps: pip install requests pandas")

    print(f"  Fetching {ADVBENCH_URL} ...")
    response = requests.get(ADVBENCH_URL, timeout=30)
    response.raise_for_status()

    # The CSV has columns: goal, target
    # We use 'goal' — the adversarial instruction itself.
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    if "goal" not in df.columns:
        sys.exit(f"Unexpected CSV columns: {df.columns.tolist()}")

    prompts = df["goal"].dropna().tolist()
    print(f"  Downloaded {len(prompts)} harmful prompts.")
    _write_lines(HARMFUL_OUT, prompts)


# ---------------------------------------------------------------------------
# XSTest — benign-aggressive (safe subset)
# ---------------------------------------------------------------------------

def download_xstest():
    print("\n[2/3] XSTest — safe (benign-aggressive) subset")
    if _already_exists(BENIGN_AGG_OUT):
        return

    try:
        import requests
        import pandas as pd
    except ImportError:
        sys.exit("Install missing deps: pip install requests pandas")

    print(f"  Fetching {XSTEST_URL} ...")
    response = requests.get(XSTEST_URL, timeout=30)
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    if "prompt" not in df.columns:
        sys.exit(f"Unexpected XSTest columns: {df.columns.tolist()}")

    # XSTest schema: id, type, prompt, note, label, contrast
    # 'type' starting with 'safe_' are the benign-aggressive safe prompts.
    # The 'label' column ('safe' / 'unsafe') is the primary filter when present.
    if "label" in df.columns:
        safe_df = df[df["label"] == "safe"]
    elif "type" in df.columns:
        safe_df = df[df["type"].str.startswith("safe_", na=False)]
    else:
        safe_df = df

    prompts: list[str] = safe_df["prompt"].dropna().tolist()
    print(f"  Found {len(prompts)} benign-aggressive prompts.")
    _write_lines(BENIGN_AGG_OUT, prompts)


# ---------------------------------------------------------------------------
# Alpaca-Cleaned — normative / safe set
# ---------------------------------------------------------------------------

def download_alpaca(n: int, seed: int):
    print(f"\n[3/3] Alpaca-Cleaned — normative set (n={n}, seed={seed})")
    # if _already_exists(NORMATIVE_OUT):
    if _is_sufficient(NORMATIVE_OUT, n):
        return

    try:
        from datasets import load_dataset, Dataset
        from typing import Any
    except ImportError:
        sys.exit("Install missing deps: pip install datasets")

    print("  Loading yahma/alpaca-cleaned from HuggingFace ...")
    ds = cast(Dataset, load_dataset("yahma/alpaca-cleaned", split="train"))

    # to_dict() is typed as returning Iterator in some HF stub versions.
    # Cast to the concrete runtime type so Pylance resolves key access.
    ds_dict = cast(dict[str, list[Any]], ds.to_dict())
    all_instructions: list[str] = ds_dict["instruction"]
    all_inputs: list[str]       = ds_dict["input"]

    filtered = []
    for instr, inp in zip(all_instructions, all_inputs):
        instr = str(instr).strip()
        # Strict Quality Filter:
        if (
            not str(inp).strip() and           # 1. No separate 'input' context 
            "\n" not in instr and              # 2. MUST be a single line (no math/lists)
            not instr.endswith(":") and        # 3. No dangling colon prompts
            len(instr) >= 20 and               # 4. Long enough to have real semantics
            len(instr) <= 250                  # 5. Not so long it's a huge outlier
        ):
            filtered.append(instr)

    print(f"  After filtering: {len(filtered)} candidate instructions.")

    if len(filtered) < n:
        print(f"  Warning: only {len(filtered)} available, using all.")
        n = len(filtered)

    random.seed(seed)
    sampled = random.sample(filtered, n)
    _write_lines(NORMATIVE_OUT, sampled)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument(
        "--normative-n", type=int, default=720,
        help="Number of Alpaca prompts to sample for the normative set (default: 720)."
    )
    parser.add_argument("--harmful-n", type=int, default=520,
                        help="Number of harmful prompts to use from AdvBench (default: 520).")      
    parser.add_argument("--benign-agg-n", type=int, default=250,
                        help="Number of benign-aggressive prompts to use (default: 250).")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for Alpaca sampling (default: 42)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Dataset Acquisition for LLM Biopsy")
    print("=" * 60)

    download_advbench()
    download_xstest()
    download_alpaca(n=args.normative_n, seed=args.seed)

    print("\nDone. All splits available under data/raw/.")
    print("  normative.txt       →", NORMATIVE_OUT)
    print("  harmful.txt         →", HARMFUL_OUT)
    print("  benign_aggressive.txt →", BENIGN_AGG_OUT)


if __name__ == "__main__":
    main()