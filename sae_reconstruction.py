# %%
import torch
from utils import load_activations
import gc
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--num_layers', type=int, default=26)
parser.add_argument('--token_types', nargs='+', type=str, default=['llm_text', 'vlm_img', 'vlm_imgtext'])
parser.add_argument('--batch_size', type=int, default=5, help='Batch size for processing activations')
args, _ = parser.parse_known_args()

num_samples = args.num_samples
num_layers = args.num_layers
token_types = args.token_types
batch_size = args.batch_size

# %%
def process_in_batches(total_samples, batch_size, process_fn):
    """
    Process data in batches to manage memory usage.
    
    Args:
        total_samples: Total number of samples to process
        batch_size: Number of samples to process in each batch
        process_fn: Function that processes a batch (start_idx, end_idx) -> results
        
    Returns:
        Combined results from all batches
    """
    results = []
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_results = process_fn(start_idx, end_idx)
        results.extend(batch_results)
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()
    return results

def load_activations_batch(computation, layers, start_idx, end_idx, model_type):
    """
    Load activations for a batch of examples.
    
    Args:
        computation: Type of computation ('sae_encoded', 'sae_decoded', or None for raw activations)
        layers: List of layer indices to load
        start_idx: Start index of the batch
        end_idx: End index of the batch
        model_type: Type of model ('llm_text', 'vlm_img', 'vlm_imgtext')
        
    Returns:
        Dictionary of activations by layer
    """
    batch_indices = list(range(start_idx, end_idx))
    
    if computation is None:
        # For raw activations
        return load_activations(
            components=['layer_output'],
            layers=layers,
            indices=batch_indices,
            model_type=model_type
        )["layer_output"]
    else:
        # For SAE encoded/decoded activations
        return load_activations(
            computation=computation,
            layers=layers,
            indices=batch_indices,
            model_type=model_type
        )

# %%
# Initialize metrics dictionaries to accumulate results
metrics_sum = {
    token_type: {
        'l2': [0] * num_layers,
        'l0': [0] * num_layers
    } for token_type in token_types
}

sample_counts = {token_type: [0] * num_layers for token_type in token_types}

# Process each token type
for token_type in token_types:
    print(f"Processing {token_type} activations...")
    
    # Process in batches
    for start_idx in tqdm(range(0, num_samples, batch_size), desc=f"Processing batches for {token_type}"):
        end_idx = min(start_idx + batch_size, num_samples)
                
        # Load original activations for this batch
        original_acts = load_activations_batch(
            computation=None,
            layers=list(range(num_layers)),
            start_idx=start_idx,
            end_idx=end_idx,
            model_type=token_type
        )
        
        # Load reconstructed activations for this batch
        reconstructed_acts = load_activations_batch(
            computation='sae_decoded',
            layers=list(range(num_layers)),
            start_idx=start_idx,
            end_idx=end_idx,
            model_type=token_type
        )
        
        # Load SAE features for this batch
        features = load_activations_batch(
            computation='sae_encoded',
            layers=list(range(num_layers)),
            start_idx=start_idx,
            end_idx=end_idx,
            model_type=token_type
        )
                
        # Calculate metrics for each layer in this batch
        for layer_idx in range(num_layers):
            if layer_idx not in original_acts or layer_idx not in reconstructed_acts or layer_idx not in features:
                print(f"  Warning: Layer {layer_idx} not found for batch {start_idx}-{end_idx-1}")
                continue
                
            # Process each sample in the batch
            for sample_idx in range(len(original_acts[layer_idx])):
                orig = original_acts[layer_idx][sample_idx][0].to('cuda')
                recon = reconstructed_acts[layer_idx][sample_idx][0].to('cuda')
                feat = features[layer_idx][sample_idx][0].to('cuda')
                
                # Calculate L2 loss
                l2_loss = (recon - orig).pow(2).mean().item()
                # Calculate L0 sparsity
                l0_loss = (feat != 0).float().mean().item()
                
                # Accumulate metrics
                metrics_sum[token_type]['l2'][layer_idx] += l2_loss
                metrics_sum[token_type]['l0'][layer_idx] += l0_loss
                sample_counts[token_type][layer_idx] += 1

                # Free memory for individual tensors
                del orig, recon, feat
        
        # Clean up batch data to free memory before loading next batch
        del original_acts, reconstructed_acts, features
        torch.cuda.empty_cache()
        gc.collect()
        
# %% Calculate final averages and structure results
print("Calculating final results...")
losses = {}

# Initialize the losses dictionary structure
if 'llm_text' in token_types:
    if 'llm' not in losses:
        losses['llm'] = {'l2': {}, 'l0': {}}
    losses['llm']['l2']['text'] = []
    losses['llm']['l0']['text'] = []

if 'vlm_img' in token_types or 'vlm_imgtext' in token_types:
    if 'vlm' not in losses:
        losses['vlm'] = {'l2': {}, 'l0': {}}
    
    if 'vlm_img' in token_types:
        losses['vlm']['l2']['image'] = []
        losses['vlm']['l0']['image'] = []
    
    if 'vlm_imgtext' in token_types:
        losses['vlm']['l2']['imgtext'] = []
        losses['vlm']['l0']['imgtext'] = []

# Calculate averages for each token type and layer
for token_type in token_types:
    for layer_idx in range(num_layers):
        if sample_counts[token_type][layer_idx] > 0:
            l2_avg = metrics_sum[token_type]['l2'][layer_idx] / sample_counts[token_type][layer_idx]
            l0_avg = metrics_sum[token_type]['l0'][layer_idx] / sample_counts[token_type][layer_idx]
            
            # Store in the appropriate place in the losses dictionary
            if token_type == 'llm_text':
                losses['llm']['l2']['text'].append(l2_avg)
                losses['llm']['l0']['text'].append(l0_avg)
            elif token_type == 'vlm_img':
                losses['vlm']['l2']['image'].append(l2_avg)
                losses['vlm']['l0']['image'].append(l0_avg)
            elif token_type == 'vlm_imgtext':
                losses['vlm']['l2']['imgtext'].append(l2_avg)
                losses['vlm']['l0']['imgtext'].append(l0_avg)
        else:
            print(f"Warning: No samples for {token_type} at layer {layer_idx}")
            # Add a placeholder value
            if token_type == 'llm_text':
                losses['llm']['l2']['text'].append(0)
                losses['llm']['l0']['text'].append(0)
            elif token_type == 'vlm_img':
                losses['vlm']['l2']['image'].append(0)
                losses['vlm']['l0']['image'].append(0)
            elif token_type == 'vlm_imgtext':
                losses['vlm']['l2']['imgtext'].append(0)
                losses['vlm']['l0']['imgtext'].append(0)

# Save the results
os.makedirs('results/vars', exist_ok=True)
torch.save(losses, 'results/vars/sae_reconstruction_losses.pt')
print("Saved losses to results/vars/sae_reconstruction_losses.pt")

# %%
def visualize_reconstruction_results(losses, 
                                    output_file='results/figures/sae_reconstruction_combined.pdf',
                                    title_l2='SAE Reconstruction Loss Across Layers',
                                    title_l0='SAE Feature Sparsity Across Layers',
                                    convergence_layer=17):
    """
    Create publication-quality line plots to visualize reconstruction results across different token types.
    
    Args:
        losses: Dictionary containing the loss metrics
        output_file: Path to save the combined figure
        title_l2: Title for the L2 loss plot
        title_l0: Title for the L0 sparsity plot
        convergence_layer: Layer to mark as convergence point
    """
    # Convert the nested dictionary to a pandas DataFrame for easier plotting
    data_l2 = []
    data_l0 = []
    
    # Process L2 loss data
    for model_type, metrics in losses.items():
        for token_type, values in metrics['l2'].items():
            for layer, value in enumerate(values):
                data_l2.append({
                    'model_type': model_type,
                    'token_type': f"{model_type}_{token_type}",
                    'layer': layer,
                    'value': value
                })
    
    # Process L0 sparsity data
    for model_type, metrics in losses.items():
        for token_type, values in metrics['l0'].items():
            for layer, value in enumerate(values):
                data_l0.append({
                    'model_type': model_type,
                    'token_type': f"{model_type}_{token_type}",
                    'layer': layer,
                    'value': value
                })
    
    # Create DataFrames
    df_l2 = pd.DataFrame(data_l2)
    df_l0 = pd.DataFrame(data_l0)
    
    # Create a mapping for renaming token types in the legend (similar to sae_descriptions_eval.py)
    token_type_mapping = {
        'vlm_image': 'VLM Image Positions',
        'vlm_imgtext': 'VLM Image-Text Positions',
        'llm_text': 'LLM Prompt Positions'
    }
    
    # Create copies of the dataframes with renamed token types for display
    df_l2_display = df_l2.copy()
    df_l0_display = df_l0.copy()
    
    df_l2_display['token_type'] = df_l2_display['token_type'].map(lambda x: token_type_mapping.get(x, x))
    df_l0_display['token_type'] = df_l0_display['token_type'].map(lambda x: token_type_mapping.get(x, x))
    
    # Set up the figure with a clean, modern style - side by side plots
    plt.figure(figsize=(20, 6))
    sns.set_style("whitegrid")
    
    # Create a custom color palette with distinct colors for each token type
    # Using a colorblind-friendly palette without red
    custom_colors = {
        'LLM Prompt Positions': '#1f77b4',       # Blue
        'VLM Image Positions': '#ff7f0e',        # Orange
        'VLM Image-Text Positions': '#2ca02c'    # Green
    }
    
    # L2 Loss plot (left)
    plt.subplot(1, 2, 1)
    _create_subplot(df_l2_display, 'value', title_l2, 'Reconstruction Loss (MSE)', custom_colors)
    
    # Mark convergence layer in L2 loss plot
    _mark_convergence_layer(df_l2, convergence_layer)
    
    # L0 Sparsity plot (right)
    plt.subplot(1, 2, 2)
    _create_subplot(df_l0_display, 'value', title_l0, 'Sparsity', custom_colors)
    
    # Mark convergence layer in L0 sparsity plot
    _mark_convergence_layer(df_l0, convergence_layer)
    
    # Improve overall appearance
    plt.tight_layout()
    
    # Save the figure with high resolution
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    plt.close()
    
    print(f"Combined visualization saved to: {output_file}")

def _create_subplot(df, y_column, title, y_label, palette):
    """
    Helper function to create a subplot with the given data.
    
    Args:
        df: DataFrame containing the data
        y_column: Column name for the y-axis values
        title: Title for the plot
        y_label: Label for the y-axis
        palette: Color palette to use (dictionary mapping token types to colors)
    """
    # Create the line plot
    ax = sns.lineplot(
        data=df,
        x='layer',
        y=y_column,
        hue='token_type',
        palette=palette,
        linewidth=2.5,
        marker='o',
        markersize=8,
        markeredgecolor='white',
        markeredgewidth=1.5
    )
    
    # Set y-axis limits with some padding
    min_y = df[y_column].min()
    max_y = df[y_column].max()
    y_padding = (max_y - min_y) * 0.05  # 5% padding
    plt.ylim(max(0, min_y - y_padding), max_y + y_padding)
    
    # Customize the plot for publication quality with increased font sizes
    plt.xlabel('Layer', fontsize=18, fontweight='bold')
    plt.ylabel(y_label, fontsize=18, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Move legend inside the figure
    plt.legend(title='Token Type', title_fontsize=18, fontsize=18, 
               frameon=True, facecolor='white', edgecolor='lightgray',
               loc='best')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure x-axis shows only every second layer value with larger font
    layers = sorted(df['layer'].unique())
    # Show only every second layer tick
    layers_to_show = layers[::2]
    # Create empty labels for the layers we want to hide
    all_labels = []
    for layer in layers:
        if layer in layers_to_show:
            all_labels.append(str(layer))
        else:
            all_labels.append('')
    
    plt.xticks(layers, all_labels, fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add subtle shading under the lines
    for token_type in df['token_type'].unique():
        token_data = df[df['token_type'] == token_type].sort_values('layer')
        plt.fill_between(
            token_data['layer'], 
            token_data[y_column], 
            alpha=0.1, 
            color=palette[token_type]
        )
    
    return ax

def _mark_convergence_layer(df, convergence_layer):
    """
    Mark the convergence region with a box and label, from the convergence layer to the end.
    
    Args:
        df: DataFrame containing the data
        convergence_layer: Starting layer for the convergence region
    """
    # Check if the convergence layer exists in the data
    if convergence_layer not in df['layer'].unique():
        print(f"Warning: Convergence layer {convergence_layer} not found in data")
        return
    
    # Get all layers and find the maximum layer
    all_layers = sorted(df['layer'].unique())
    max_layer = max(all_layers)
    
    # Get data for the convergence region (from convergence_layer to the end)
    convergence_region_data = df[df['layer'] >= convergence_layer]
    
    # Get the y-axis limits
    y_min, y_max = plt.ylim()
    y_range = y_max - y_min
    
    # Calculate the box dimensions
    values = convergence_region_data['value'].values
    min_value = min(values) - 0.05 * y_range
    max_value = max(values) + 0.05 * y_range
    
    # Draw a box around the convergence region points
    box_width = (max_layer - convergence_layer) / 2 + 0.5  # Width of the box
    box_center = (max_layer + convergence_layer) / 2  # Center of the box
    
    # Use purple instead of red for the convergence box
    convergence_color = '#9467bd'  # Purple
    
    rect = plt.Rectangle(
        (convergence_layer - 0.5, min_value),  # (x, y) of bottom left corner
        max_layer - convergence_layer + 1,  # width (including both end layers)
        max_value - min_value,  # height
        linewidth=2,
        edgecolor=convergence_color,
        facecolor='none',
        linestyle='-',
        zorder=5,
        alpha=0.8
    )
    plt.gca().add_patch(rect)
    
    # Add a "convergence" label at the top of the box
    plt.text(
        x=box_center,
        y=max_value + 0.01 * y_range,  # Position slightly above the box
        s="convergence",
        color=convergence_color,
        fontsize=18,
        fontweight='bold',
        ha='center',
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor=convergence_color, pad=2, boxstyle='round,pad=0.5')
    )

# Create directory if it doesn't exist
losses = torch.load('results/vars/sae_reconstruction_losses.pt')

# Visualize the results using the current losses (no need to reload)
visualize_reconstruction_results(
    losses,
    output_file='results/figures/sae_reconstruction_combined.pdf',
    title_l2='SAE Reconstruction Loss Across Layers',
    title_l0='SAE Feature Sparsity Across Layers',
    convergence_layer=18
)

print("Visualization saved to: results/figures/sae_reconstruction_combined.pdf")

# %%

# %%
