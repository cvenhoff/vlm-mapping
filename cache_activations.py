# %%
import dotenv
import os
dotenv.load_dotenv(".env")

from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch
from tqdm import tqdm
from data.llava_dataset import GemmaLLaVADataset 
from transformers import CLIPImageProcessor, CLIPVisionModel
import argparse

# %% config
parser = argparse.ArgumentParser()
parser.add_argument('--cache_n_examples', type=int, default=50)
parser.add_argument('--output_dir', type=str, default='activations')
parser.add_argument('--projector_weights', type=str, default='train/weights/projector_gemma-2-2b-it_finetune.pth')
parser.add_argument('--json_file', type=str, default='data/LLaVA-Instruct/llava_v1_5_mix665k.json')
parser.add_argument('--image_dir', type=str, default='data/LLaVA-Instruct/')
args, _ = parser.parse_known_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# %%
saes = []
for layer in tqdm(range(26), desc="Loading SAEs"):
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{layer}/width_16k/canonical",
        device="cuda",
    )
    saes.append(sae)

# %% load models and saes
model = HookedTransformer.from_pretrained("gemma-2-2b-it").to("cuda")
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# %%
projector_state_dict = torch.load(args.projector_weights)
projector = torch.nn.Linear(vision_model.config.hidden_size, model.cfg.d_model).to("cuda")
projector.load_state_dict(projector_state_dict)

# %% load image and prepare image hook
dataset = GemmaLLaVADataset(json_file=args.json_file,
                            image_dir=args.image_dir,
                            tokenizer=model.tokenizer,
                            image_processor=image_processor)

# %%
def get_image_hook(image_features):
    def image_hook(act, hook):
        if act.shape[1] > 1:
            act[:, 1:257] = image_features
        return act
    return image_hook

# Process examples
for dataset_id in tqdm(range(min(len(dataset), args.cache_n_examples)), desc="Processing examples"):
    try:
        # Get example data
        input_ids = dataset[dataset_id]["input_ids"].to("cuda")
        assistant_positions = dataset[dataset_id]["assistant_positions"][0]
        answer_clue = model.tokenizer.decode(input_ids[assistant_positions[0]:assistant_positions[1]-1])
        image_tensor = dataset[dataset_id]["image_tensor"]
        
        # Process image
        image = vision_model(image_tensor.to("cuda"))
        image_features = projector(image.last_hidden_state)
        image_features = image_features.detach()[:, 1:]
        
        # Prepare input
        prompt = model.tokenizer.decode(input_ids[1:]).split("<start_of_turn>model")[0] + "<start_of_turn>model\n"
        text_prompt = (model.tokenizer.decode(input_ids[1:]).split("<start_of_turn>model")[0] + "<start_of_turn>model\n").replace("<image>", "Consider the following information: " + answer_clue + " ")

        input_ids = model.tokenizer.encode(prompt, return_tensors='pt').to("cuda")[0]
        input_ids = torch.cat([input_ids[:1], torch.zeros(256, dtype=torch.int64).to("cuda"), input_ids[1:]], dim=0)

        text_input_ids = model.tokenizer.encode(text_prompt, return_tensors='pt').to("cuda")[0]

        # Run model with cache
        with model.hooks(fwd_hooks=[('blocks.0.hook_resid_pre', get_image_hook(image_features))]):
            _, cache = model.run_with_cache(input_ids.unsqueeze(0))

        _, text_cache = model.run_with_cache(text_input_ids.unsqueeze(0))
                
        projector_outputs = cache[f'blocks.0.hook_resid_pre'][:, 1:257, :]

        example_data = {'projector_outputs': projector_outputs.cpu()}

        for layer_idx, sae in enumerate(saes):
            # Get layer-specific activations
            layer_image_acts = cache[f'blocks.{layer_idx}.hook_resid_post'][:, 1:257, :]
            layer_image_text_acts = cache[f'blocks.{layer_idx}.hook_resid_post'][:, 257:, :]
            layer_text_acts = text_cache[f'blocks.{layer_idx}.hook_resid_post'][:, 1:, :]

            # Encode and decode
            sae_image_acts = sae.encode(layer_image_acts)
            sae_image_text_acts = sae.encode(layer_image_text_acts)
            sae_text_acts = sae.encode(layer_text_acts)

            sae_image_recon = sae.decode(sae_image_acts)
            sae_image_text_recon = sae.decode(sae_image_text_acts)
            sae_text_recon = sae.decode(sae_text_acts)
            
            # Store in dictionary
            example_data[f'layer_{layer_idx}'] = {
                'image_activations': layer_image_acts.cpu(),
                'image_text_activations': layer_image_text_acts.cpu(),
                'text_activations': layer_text_acts.cpu(),
                'sae_image_activations': sae_image_acts.cpu(),
                'sae_image_text_activations': sae_image_text_acts.cpu(),
                'sae_text_activations': sae_text_acts.cpu(),
                'sae_image_reconstructions': sae_image_recon.cpu(),
                'sae_image_text_reconstructions': sae_image_text_recon.cpu(),
                'sae_text_reconstructions': sae_text_recon.cpu()
            }
        
        # Save activations
        save_path = os.path.join(args.output_dir, f'example_{dataset_id}.pt')
        torch.save(example_data, save_path)
        
        # Clear cache
        del cache
        del example_data
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing example {dataset_id}: {str(e)}")
        continue

# %%
