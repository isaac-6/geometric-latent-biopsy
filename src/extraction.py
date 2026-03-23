import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional

class LatentExtractor:
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initializes the model and tokenizer for latent extraction.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"Loading tokenizer {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading model {model_name} to {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            dtype=torch.float16 if "cuda" in self.device else torch.float32
        )
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers

    @torch.no_grad()
    def get_last_token_activations(self, prompt: str) -> torch.Tensor:
        """
        Runs a forward pass and extracts the hidden states of the final token for all layers.
        
        Returns:
            torch.Tensor of shape (num_layers, hidden_dimension) 
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # output_hidden_states=True forces HF to return the residual stream at each layer
        outputs = self.model(
            **inputs, 
            output_hidden_states=True, 
            return_dict=True
        )
        
        # hidden_states is a tuple of (embedding_layer + num_layers)
        # We drop the embedding layer [0] and stack the rest
        hidden_states = outputs.hidden_states[1:] 
        
        # Extract the last token's activation across all layers
        # hidden_states[i] shape: (batch_size=1, seq_len, hidden_dim)
        last_token_activations = [layer_states[0, -1, :] for layer_states in hidden_states]
        
        # Stack into a single tensor: shape (num_layers, hidden_dim)
        return torch.stack(last_token_activations)