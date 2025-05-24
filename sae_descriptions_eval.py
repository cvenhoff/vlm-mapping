# %%
import dotenv
dotenv.load_dotenv(".env")

from utils import load_activations
from data.llava_dataset import SimpleLLaVADataset 
import http.client
import json
import time
import numpy as np
from openai import OpenAI
import os
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from tqdm import tqdm
import gc
import argparse
import sys

# %% config
parser = argparse.ArgumentParser()
parser.add_argument('--n_features_per_layer', type=int, default=3, help='Number of features per layer to evaluate')
parser.add_argument('--num_freq_features', type=int, default=500, help='Number of samples to compute feature frequencies from')
parser.add_argument('--frequency_threshold', type=float, default=0.05, help='Maximum allowed frequency for a feature')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
parser.add_argument('--num_samples', type=int, default=10, help='Number of examples to process')
parser.add_argument('--projector_weights', type=str, default='train/weights/projector_gemma-2-2b-it_finetune.pth')
parser.add_argument('--json_file', type=str, default='data/LLaVA-Instruct/llava_v1_5_mix665k.json')
parser.add_argument('--image_dir', type=str, default='data/LLaVA-Instruct/')
parser.add_argument('--token_types', nargs='+', type=str, default=['llm_text', 'vlm_img'])

# Check if running in Jupyter and handle arguments accordingly
if any(arg.endswith('.json') and '/jupyter/' in arg for arg in sys.argv):
    # Running in Jupyter - use default arguments
    args = parser.parse_args([])
else:
    # Normal command-line execution
    args, _ = parser.parse_known_args()

token_types = args.token_types

# %%
dataset = SimpleLLaVADataset(json_file=args.json_file,
                            image_dir=args.image_dir)

# %%
neuronpedia_cache = {}
def get_feature_information(layer, feature, timeout=5):
    try:
        if (layer, feature) in neuronpedia_cache:
            return neuronpedia_cache[(layer, feature)]
        
        # give me the 128-step interval that the feature is in
        interval = feature // 128
        interval_start = interval * 128
        interval_end = (interval + 1) * 128
        json_path = f"data/gemma-2-2b-descriptions/{layer}-gemmascope-res-16k/{interval_start}-{interval_end}.json"
        with open(json_path, 'r') as f:
            data = json.load(f)

        feature_datapoint = data[feature - interval_start]
        description = feature_datapoint["explanations"][0]["description"]
        density = feature_datapoint["frac_nonzero"]

        neuronpedia_cache[(layer, feature)] = [density, description]

        return neuronpedia_cache[(layer, feature)]
    
    except Exception as e:
        print(f"Error: {e}")
        return [None, None]

# %%
def chat(prompt, image=None):
    client = OpenAI()
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            base64_image = None
            if image is not None:
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Decode bytes to string

            if base64_image is not None:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                temperature=0.001,
                )
                return response.choices[0].message.content
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Failed after {max_retries} attempts. Error: {e}")
                raise
            time.sleep(20)  # 0 second sleep between retries

# %%
def evaluate_sae_features_batch(feature_descriptions, image):
    # Combine multiple feature descriptions into a single prompt
    descriptions_text = "\n".join([f"Feature {i+1} description: '{desc}'" for i, desc in enumerate(feature_descriptions)])
    
    prompt = f"""Analyze this image and determine if ANY of the provided SAE (Sparse Autoencoder) feature descriptions match a key concept in the image.

{descriptions_text}

First, write a brief description of all objects, activities, high-level concepts, etc. that are shown in the image.
Then, determine if ANY of the feature descriptions matches one of these.

Return your response in exactly this format:
Image description: [1-2 sentences describing the objects, activities, high-level concepts, etc. in the image]

Match found: [YES or NO]
Explanation: [1 sentence explaining which feature(s) matched the image description, if any, or why none matched]"""
    
    response = chat(prompt, image=image)
    
    try:
        # Parse the response
        match_line = [line for line in response.split('\n') if line.startswith('Match found:')][0]
        match_result = 1 if 'YES' in match_line.upper() else 0
        explanation_line = [line for line in response.split('\n') if line.startswith('Explanation:')][0]
        explanation = explanation_line.split(':')[1].strip()
        
        return match_result, explanation
        
    except:
        # If parsing fails, return 0 and empty explanation
        return 0, "Failed to parse response"

# %%
def process_in_batches(total_samples, batch_size, process_fn):
    results = []
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_results = process_fn(start_idx, end_idx)
        results.extend(batch_results)
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()
    return results

# Calculate feature frequencies in batches
if os.path.exists(f'results/vars/feature_frequencies_{args.num_freq_features}.pt'):
    feature_frequencies = torch.load(f'results/vars/feature_frequencies_{args.num_freq_features}.pt')
    frequency_counts = torch.load(f'results/vars/frequency_counts_{args.num_freq_features}.pt')
else:
    feature_frequencies = {m: [None for _ in range(26)] for m in token_types}
    frequency_counts = {m: [0 for _ in range(26)] for m in token_types}

    for m in token_types:
        for start_idx in tqdm(range(0, args.num_freq_features, args.batch_size), desc=f"Computing feature frequencies for {m}"):
            end_idx = min(start_idx + args.batch_size, args.num_freq_features)
            batch_indices = list(range(start_idx, end_idx))
            
            sae_activations = load_activations(
                computation='sae_encoded', 
                layers=list(range(26)),
                indices=batch_indices,
                model_type=m
            )
            
            for layer_idx in range(26):
                # Stack activations and compute binary activation mask
                stacked_activations = torch.cat([x.mean(dim=1) for x in sae_activations[layer_idx]], dim=0)
                binary_activations = (stacked_activations > 0).float().mean(dim=0, keepdim=True)
                
                # Compute frequency across samples for each feature
                feature_freq = binary_activations
                
                if feature_frequencies[m][layer_idx] is None:
                    feature_frequencies[m][layer_idx] = feature_freq
                else:
                    feature_frequencies[m][layer_idx] += feature_freq
                frequency_counts[m][layer_idx] += 1
            
            del sae_activations
            torch.cuda.empty_cache()
            gc.collect()

    # Normalize frequencies
    for m in token_types:
        for layer_idx in range(26):
            feature_frequencies[m][layer_idx] /= frequency_counts[m][layer_idx]

    torch.save(feature_frequencies, f'results/vars/feature_frequencies_{args.num_freq_features}.pt')
    torch.save(frequency_counts, f'results/vars/frequency_counts_{args.num_freq_features}.pt')

# %% Dictionary to store results
results = {
    'layer': [],
    'token_type': [],
    'match_rate': []
}

# Dictionary to store detailed results
detailed_results = {
    'evaluations': []
}

for token_type in token_types:
    print(f"Evaluating {token_type} features")
    progress_bar = tqdm(total=args.num_samples, desc=f"Token Type: {token_type}.")
    for batch_start in range(0, args.num_samples, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.num_samples)
        batch_indices = list(range(batch_start, batch_end))
        
        # Load batch activations
        sae_activations = load_activations(
            computation='sae_encoded',
            layers=list(range(26)),
            indices=batch_indices,
            model_type=token_type
        )
        
        for sample_idx in range(len(batch_indices)):
            image = dataset[batch_indices[sample_idx]][0]

            for layer_idx in range(26):
                # Get activations for current sample and layer
                activations = sae_activations[layer_idx][sample_idx][0].to("cuda")
                
                # Zero out activations for high-frequency features
                freq_mask = (feature_frequencies[token_type][layer_idx].to("cuda") <= args.frequency_threshold)
                activations = activations * freq_mask
                
                # Get top features using filtered activations
                flat_activations = activations.float().flatten().detach()
                top_indices = flat_activations.argsort(descending=True)

                batch_descriptions = []
                batch_feature_indices = []
                top_idx = 0

                while len(batch_descriptions) < args.n_features_per_layer:
                    feature_idx_flat = top_indices[top_idx]
                    feature_idx = feature_idx_flat % activations.shape[1]
                    token_idx = feature_idx_flat // activations.shape[1]
                    top_idx += 1

                    density, sae_description = get_feature_information(
                        layer=layer_idx,
                        feature=feature_idx.item()
                    )

                    if density is None or sae_description is None:
                        continue

                    if density > 0.005:
                        continue

                    if feature_idx.item() in batch_feature_indices:
                        continue
                                            
                    if sae_description:
                        batch_descriptions.append(sae_description)
                        batch_feature_indices.append(feature_idx.item())

                if batch_descriptions:
                    match_result, explanation = evaluate_sae_features_batch(batch_descriptions, image)
                    
                    layer_evaluations = {
                        'layer': layer_idx,
                        'token_type': token_type,
                        'feature_indices': batch_feature_indices,
                        'feature_descriptions': batch_descriptions,
                        'match_found': match_result,
                        'explanation': explanation
                    }
                    detailed_results['evaluations'].append(layer_evaluations)
                    
                    results['layer'].append(layer_idx)
                    results['token_type'].append(token_type)
                    results['match_rate'].append(match_result)

            progress_bar.update(1)
        
        # Clean up batch memory
        del sae_activations
        torch.cuda.empty_cache()
        gc.collect()

# Save detailed results to JSON
os.makedirs('results/vars', exist_ok=True)
with open('results/vars/sae_feature_evaluations.json', 'w') as f:
    json.dump(detailed_results, f, indent=2)

# Convert results to DataFrame and process for visualization
df = pd.DataFrame(results)

# Average match rates across samples for each layer and token type
df_avg = df.groupby(['layer', 'token_type'])['match_rate'].mean().reset_index()
 
# Create bar chart visualization
plt.figure(figsize=(20, 8))

# Set the width of each bar and positions of the bars
bar_width = 0.25
layers = sorted(df_avg['layer'].unique())
x = np.arange(len(layers))

# Plot bars for each token type
for i, token_type in enumerate(token_types):
    token_data = df_avg[df_avg['token_type'] == token_type]
    token_data = token_data.sort_values('layer')
    plt.bar(x + i*bar_width, 
            token_data['match_rate'], 
            bar_width, 
            label=token_type,
            alpha=0.8)

# Customize the plot
plt.xlabel('Layer')
plt.ylabel('Match Rate')
plt.title('SAE Feature Key Concept Match Rate Across Layers\n(Averaged across samples)')
plt.xticks(x + bar_width, layers)
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for i, token_type in enumerate(token_types):
    token_data = df_avg[df_avg['token_type'] == token_type]
    token_data = token_data.sort_values('layer')
    for j, v in enumerate(token_data['match_rate']):
        plt.text(j + i*bar_width, v + 0.02, 
                f'{v:.2f}', 
                ha='center', 
                fontsize=8,
                rotation=90)

plt.tight_layout()
plt.savefig('results/figures/sae_feature_match_rate_bars.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Save both raw and processed data
torch.save(df, 'results/vars/sae_feature_match_rate_raw.pt')
torch.save(df_avg, 'results/vars/sae_feature_match_rate_averaged.pt')


# %%
def visualize_results(results_file='results/vars/sae_feature_match_rate_averaged.pt', 
                     token_types=None, 
                     output_file='results/figures/sae_feature_match_rate_lines.png',
                     title='SAE Feature Key Concept Match Rate Across Layers',
                     convergence_layer=17):
    """
    Create publication-quality line plots to visualize results across different token types.
    
    Args:
        results_file: Path to the saved results file (default: 'results/vars/sae_feature_match_rate_averaged.pt')
        token_types: List of token types to include in visualization (default: None, which uses all available)
        output_file: Path to save the output figure (default: 'results/figures/sae_feature_match_rate_lines.png')
        title: Title for the plot (default: 'SAE Feature Key Concept Match Rate Across Layers')
        convergence_layer: Layer to mark as convergence point (default: 17)
    """
    # Load the results
    df_avg = torch.load(results_file, weights_only=False)
    
    # Filter token types if specified
    if token_types is not None:
        df_avg = df_avg[df_avg['token_type'].isin(token_types)]
    
    # Create a mapping for renaming token types in the legend
    token_type_mapping = {
        'llm_text': 'LLM Prompt Positions',
        'vlm_img': 'VLM Image Positions'
    }
    
    # Create a copy of the dataframe with renamed token types for display
    df_display = df_avg.copy()
    df_display['token_type'] = df_display['token_type'].map(lambda x: token_type_mapping.get(x, x))
    
    # Set up the figure with a clean, modern style - slightly taller than before
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    # Create a custom color palette with distinct colors for each token type
    # Using distinct colors that are easily distinguishable but don't include red
    custom_colors = {
        'LLM Prompt Positions': '#1f77b4',    # Blue
        'VLM Image Positions': '#ff7f0e',     # Orange
    }
    
    # Create the line plot with renamed token types
    ax = sns.lineplot(
        data=df_display,
        x='layer',
        y='match_rate',
        hue='token_type',
        palette=custom_colors,
        linewidth=2.5,
        marker='o',
        markersize=8,
        markeredgecolor='white',
        markeredgewidth=1.5
    )
    
    # Set y-axis limit to 1 and maintain the lower limit close to the data
    min_y = df_avg['match_rate'].min()
    y_padding = min_y * 0.05  # 5% padding for lower limit
    plt.ylim(min_y - y_padding, 1.05)
    
    # Customize the plot for publication quality with increased font sizes
    plt.xlabel('Layer', fontsize=18, fontweight='bold')
    plt.ylabel('Match Rate', fontsize=18, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Move legend inside the figure
    plt.legend(title='Token Type', title_fontsize=18, fontsize=18, 
               frameon=True, facecolor='white', edgecolor='lightgray',
               loc='upper left')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure x-axis shows integer values for layers with larger font
    layers = sorted(df_avg['layer'].unique())
    # Show ticks every 2 layers
    layers_to_show = [layer for layer in layers if layer % 2 == 0]
    plt.xticks(layers_to_show, fontsize=18)
    plt.yticks(fontsize=18)
    
    # Add subtle shading under the lines
    for token_type in df_display['token_type'].unique():
        # Get the original token type name for filtering
        original_token_type = next(k for k, v in token_type_mapping.items() if v == token_type) if token_type in token_type_mapping.values() else token_type
        token_data = df_avg[df_avg['token_type'] == original_token_type].sort_values('layer')
        plt.fill_between(
            token_data['layer'], 
            token_data['match_rate'], 
            alpha=0.1, 
            color=custom_colors[token_type]
        )
    
    # Improve overall appearance
    plt.tight_layout()
    
    # Save the figure with high resolution
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Mark convergence layer
    _mark_convergence_layer(df_avg, convergence_layer)
    
    # Save the figure with high resolution again after adding convergence box
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    plt.close()
    
    print(f"Visualization saved to: {output_file}")

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
    values = convergence_region_data['match_rate'].values
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

visualize_results(
    results_file='results/vars/sae_feature_match_rate_averaged.pt',
    token_types=['llm_text', 'vlm_img'],
    output_file='results/figures/sae_feature_match_rate_lines.pdf',
    title='SAE Feature Alignment Across Layers',
    convergence_layer=18
)
# %%
