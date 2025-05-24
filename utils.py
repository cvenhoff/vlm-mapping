import torch
import numpy as np
from pathlib import Path

def load_activations(
    computation=None,
    components=None,
    layers=None,
    indices=None,
    base_dir='activations',
    model_type='vlm_imgtext'
):
    base_path = Path(base_dir)
    
    # Handle SAE computations (encoded/decoded)
    if computation in ['sae_encoded', 'sae_decoded']:
        result = {layer: [] for layer in layers}
        
        for idx in indices:
            file_path = base_path / f'example_{idx}.pt'
            if not file_path.exists():
                print(f"Warning: File not found for example {idx}")
                continue
                
            data = torch.load(file_path, weights_only=False)
            
            for layer in layers:
                if layer >= len(data):
                    print(f"Warning: Layer {layer} not found in example {idx}")
                    continue
                
                # Extract the appropriate activations based on computation type
                if computation == 'sae_encoded':
                    if model_type == 'vlm_img':
                        acts = data[f'layer_{layer}']['sae_image_activations']
                    elif model_type == 'vlm_imgtext':
                        acts = data[f'layer_{layer}']['sae_image_text_activations']
                    elif model_type == 'llm_text':
                        acts = data[f'layer_{layer}']['sae_text_activations']
                else:  # sae_decoded
                    if model_type == 'vlm_img':
                        acts = data[f'layer_{layer}']['sae_image_reconstructions']
                    elif model_type == 'vlm_imgtext':
                        acts = data[f'layer_{layer}']['sae_image_text_reconstructions']
                    elif model_type == 'llm_text':
                        acts = data[f'layer_{layer}']['sae_text_reconstructions']

                result[layer].append(acts.to("cuda"))
        
        return result

    elif computation == 'projector_outputs':

        results = []
        for idx in indices:
            file_path = base_path / f'example_{idx}.pt'
            if not file_path.exists():
                print(f"Warning: File not found for example {idx}")
                continue
            
            data = torch.load(file_path, weights_only=False)
            results.append(data['projector_outputs'].to("cuda"))
        
        return results
    
    # Handle raw activations
    else:
        result = {comp: {layer: [] for layer in layers} for comp in components}
        
        for idx in indices:
            file_path = base_path / f'example_{idx}.pt'
            if not file_path.exists():
                print(f"Warning: File not found for example {idx}")
                continue
                
            data = torch.load(file_path, weights_only=False)
            
            for comp in components:
                for layer in layers:
                    if layer >= len(data):
                        print(f"Warning: Layer {layer} not found in example {idx}")
                        continue
                    
                    # For raw activations, use the image_activations as layer output
                    if model_type == 'vlm_img':
                        acts = data[f'layer_{layer}']['image_activations']
                    elif model_type == 'vlm_imgtext':
                        acts = data[f'layer_{layer}']['image_text_activations']
                    elif model_type == 'llm_text':
                        acts = data[f'layer_{layer}']['text_activations']
                    result[comp][layer].append(acts.to("cuda"))
        
        return result
