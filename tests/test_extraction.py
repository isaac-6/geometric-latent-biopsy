import torch
from src.extraction import LatentExtractor

def test_latent_extractor():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # 1. Initialize extractor
    extractor = LatentExtractor(model_name=model_name)
    
    # 2. Check device assignment
    assert extractor.device == "cuda" if torch.cuda.is_available() else "cpu", "Device assignment failed"
    
    # 3. Define a harmless test prompt
    prompt = "Translate the following English text to French: 'Hello, how are you?'"
    
    # 4. Extract activations
    activations = extractor.get_last_token_activations(prompt)
    
    # 5. Verify mathematical shapes
    num_layers = extractor.model.config.num_hidden_layers
    hidden_dim = extractor.model.config.hidden_size
    
    print(f"\nExtracted shape: {activations.shape}")
    print(f"Expected shape: ({num_layers}, {hidden_dim})")
    
    assert isinstance(activations, torch.Tensor), "Output must be a PyTorch Tensor"
    assert activations.shape == (num_layers, hidden_dim), "Activation shape mismatch"
    assert not torch.isnan(activations).any(), "NaN values found in activations"
    
    print("✅ All extraction tests passed successfully!")

if __name__ == "__main__":
    test_latent_extractor()