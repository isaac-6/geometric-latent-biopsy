import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extraction import LatentExtractor        # type: ignore[import-untyped]
from theta import compute_theta_core          # type: ignore[import-untyped]

def main():
    print("--- Starting Latent Topology Analysis ---")
    os.makedirs("results/figures", exist_ok=True)
    
    extractor = LatentExtractor("Qwen/Qwen2.5-0.5B-Instruct")
    
    # 20 diverse prompts covering coding, QA, creative writing, and translation
    diverse_prompts =[
        # Factual QA
        "What is the capital of France?", "Who wrote Romeo and Juliet?", "Explain the water cycle.", "What is photosynthesis?", "How many continents are there?",
        # Creative Writing
        "Write a poem about a robot.", "Draft a story about a space explorer.", "Compose a haiku about autumn.", "Write a funny joke.", "Describe a beautiful sunset.",
        # Coding / Logic
        "Write a Python function to sort a list.", "What is an object-oriented programming language?", "Solve this math problem: 2+2*3.", "Explain recursion.", "How does a database index work?",
        # Translation / Linguistics
        "Translate 'Good morning' to Spanish.", "What does the word 'ephemeral' mean?", "How do you say 'Thank you' in Japanese?", "Summarize the rules of English grammar.", "Give me a list of 5 synonyms for 'happy'."
    ]
    
    print(f"\n1. Extracting activations for {len(diverse_prompts)} diverse prompts...")
    acts = torch.stack([extractor.get_last_token_activations(p) for p in diverse_prompts])
    # acts shape: (num_prompts, num_layers, hidden_dim)
    
    num_prompts, num_layers, hidden_dim = acts.shape
    
    mean_thetas = []
    std_thetas =[]
    
    print("\n2. Computing pairwise intra-class angular distances...")
    # Analyze each layer independently
    for layer in range(num_layers):
        layer_acts = acts[:, layer, :] # shape: (num_prompts, hidden_dim)
        
        # We compute the pairwise Theta between all prompts
        # To do this efficiently, we expand dims
        acts_expanded_1 = layer_acts.unsqueeze(1) # (N, 1, D)
        acts_expanded_2 = layer_acts.unsqueeze(0) # (1, N, D)
        
        # Compute Theta between all pairs using our verified core math
        pairwise_theta_matrix = compute_theta_core(acts_expanded_1, acts_expanded_2)
        
        # Extract upper triangle (excluding self-pairs which are 0.0)
        triu_indices = torch.triu_indices(num_prompts, num_prompts, offset=1)
        pairwise_distances = pairwise_theta_matrix[triu_indices[0], triu_indices[1]]
        
        mean_thetas.append(pairwise_distances.mean().item())
        std_thetas.append(pairwise_distances.std().item())

    # 3. Plotting the results
    print("\n3. Plotting structural variance...")
    layers = list(range(num_layers))
    mean_thetas = np.array(mean_thetas)
    std_thetas = np.array(std_thetas)
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, mean_thetas, label="Mean Pairwise Theta", color="blue", marker="o")
    plt.fill_between(layers, mean_thetas - std_thetas, mean_thetas + std_thetas, color="blue", alpha=0.2, label="±1 Std Dev")
    
    # Reference line: Orthogonal vectors in high-dim space have a Theta of pi/2 (~1.57)
    plt.axhline(y=np.pi/2, color="red", linestyle="--", label="Orthogonal (Random) Baseline")
    
    plt.xlabel("Layer")
    plt.ylabel("Pairwise Angular Distance (Radians)")
    plt.title("Latent Topology: How scattered are safe prompts?")
    plt.legend()
    plt.grid(True)
    
    out_path = "results/figures/latent_topology.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved topology analysis to {out_path}")

if __name__ == "__main__":
    main()