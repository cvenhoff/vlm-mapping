# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformer_lens import HookedTransformer
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.llava_dataset import GemmaLLaVADataset
import argparse
from dotenv import load_dotenv
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
import itertools

load_dotenv(dotenv_path='../.env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN') 
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')

# 
# list files in current directory
print(os.listdir())

# %%
def get_default_config():
    """Returns a dictionary with default configuration parameters"""
    return {
        # Training parameters
        'batch_size': 2,
        'n_batches': 100000000,
        'n_epochs': 1,
        'initial_learning_rate': 1e-3,
        'min_learning_rate': 2e-5,
        'warmup_ratio': 0.03,
        'name': 'pretrain',
        'json_file': '../data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json',
        'image_dir': '../data/LLaVA-Pretrain/images/',
        'language_model': 'gemma-2-2b-it',
        'd_type': 'bfloat16',
        'max_length': 2048,
        'save_every': 100,
        'gradient_accumulation_steps': 32,
    }

# %%
def get_args():
    parser = argparse.ArgumentParser()
    defaults = get_default_config()
    for key, value in defaults.items():
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true')
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)
    
    # Only parse known args to avoid conflicts with Jupyter's arguments
    args, unknown = parser.parse_known_args()
    return vars(args)

# Modify the config selection logic
def get_config():
    # Check if running in Jupyter
    if any(x in sys.modules for x in ['ipykernel', 'IPython']):
        return get_default_config()
    else:
        return get_args()

# Use the new function to get config
config = get_config()

# Initialize accelerator with gradient accumulation
accelerator = Accelerator()

device = accelerator.device
set_seed(475)  # Set seed for reproducibility

# %%
# Define the path for saving and loading the projector weights
PROJECTOR_WEIGHTS_PATH = f'weights/projector_{config["language_model"]}_{config["name"]}'

# %%
# Initialize models and processors
vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

language_model = HookedTransformer.from_pretrained_no_processing(
    config['language_model'], 
    dtype=config['d_type']
)

tokenizer = language_model.tokenizer
    
# %%
class LLaVAModel(nn.Module):
    def __init__(self, vision_model, language_model, tokenizer, projection_dim=2304):
        super().__init__()

        self.projection_dim = language_model.cfg.d_model

        self.vision_model = vision_model
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.projection = self.build_vision_projector(
            vision_model.config.hidden_size,
            self.projection_dim,
        )

        # Freeze vision model parameters
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Freeze language model parameters
        for param in self.language_model.parameters():
            param.requires_grad = False


    def build_vision_projector(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def load_projector_weights(self, path):
        path_ = path + '.pth'
        if os.path.exists(path_):
            d = torch.load(path_)
            if('module.weight' in d):
                d['weight'] = d['module.weight']
                d['bias'] = d['module.bias']
                del d['module.weight']
                del d['module.bias']
            self.projection.load_state_dict(d)
            print(f"Loaded projector weights from {path}")
        else:
            print(f"No projector weights found at {path}")

    def save_projector_weights(self, path):
        torch.save(self.projection.state_dict(), path)

    def forward(self, image, input_ids, attention_mask):
        vision_outputs = self.vision_model(image)
        image_features = self.projection(vision_outputs.last_hidden_state[:,1:])
      
        input_embeds = self.language_model.embed(input_ids)

        insert_position = 1  # Adjust if needed based on your special tokens

        prefix_embeds = input_embeds[:, :insert_position]
        suffix_embeds = input_embeds[:, insert_position:]

        combined_embeds = torch.cat([prefix_embeds, image_features, suffix_embeds], dim=1)
        
        prefix_attention = attention_mask[:, :insert_position]
        suffix_attention = attention_mask[:, insert_position:]
        vision_attention = torch.ones(attention_mask.shape[0], image_features.shape[1], device=attention_mask.device)
        combined_attention_mask = torch.cat([prefix_attention, vision_attention, suffix_attention], dim=1).to(torch.int64)

        x = combined_embeds
        logs = {}  # Initialize dictionary for all wandb logs
        
        for i, block in enumerate(self.language_model.blocks):
            x = block(x, attention_mask=combined_attention_mask)
            
            # Calculate per-position and average residual norms
            resid_norms = x.norm(dim=-1).mean(0)  # [seq_len]
            avg_resid_norm = resid_norms.mean()      

        x = self.language_model.ln_final(x)
        logits = x @ self.language_model.W_U
        
        return logits
# Create LLaVA model
llava_model = LLaVAModel(vision_model, language_model, tokenizer)

# Load projector weights if available
llava_model.load_projector_weights(PROJECTOR_WEIGHTS_PATH)

llava_model = llava_model.to(device)

# %%
# Create dataset and dataloader
dataset = GemmaLLaVADataset(json_file=config['json_file'],
                            image_dir=config['image_dir'],
                            tokenizer=tokenizer,
                            image_processor=image_processor)

def collate_fn(batch):
    # Stack images properly
    images = torch.cat([item['image_tensor'] for item in batch], dim=0)
    
    # Get sequences and their lengths
    sequences = [item['input_ids'] for item in batch]
    assistant_positions = [item['assistant_positions'] for item in batch]
    
    # Get max length, but cap it at config['max_length']
    max_len = min(max(seq.size(0) for seq in sequences), config['max_length'])
    
    # Pad and truncate sequences
    padded_sequences = torch.stack([
        torch.cat([
            seq[:max_len],  # Truncate if longer than max_len
            torch.full((max(0, max_len - seq.size(0)),), tokenizer.pad_token_id, dtype=torch.long)
        ]) for seq in sequences
    ])
    
    # Adjust assistant positions for truncation
    truncated_positions = []
    for positions in assistant_positions:
        valid_positions = []
        for start, end in positions:
            if start < max_len - 1:  # -1 to ensure at least one token can be predicted
                valid_positions.append((start, min(end, max_len)))
        truncated_positions.append(valid_positions)
    
    # Create attention masks
    attention_mask = (padded_sequences != tokenizer.pad_token_id).long()
    
    return {
        "image_tensor": images,
        "input_ids": padded_sequences,
        "attention_mask": attention_mask,
        "assistant_positions": truncated_positions,
        "image_idx": [item['image_idx'] for item in batch]
    }

# Modify dataloader creation
dataloader = DataLoader(
    dataset, 
    batch_size=config['batch_size'], 
    collate_fn=collate_fn,
    shuffle=True
)

# %%
# Initialize wandb before your training loop
if accelerator.is_main_process:
    wandb.init(project="gemma_clip", config=config)
    
# Initialize the optimizer with the initial learning rate from config
optimizer = optim.AdamW(llava_model.projection.parameters(), lr=config['initial_learning_rate'])

# Calculate the total number of steps and warmup steps
total_steps = config['n_epochs'] * min(config['n_batches'], len(dataloader))
warmup_steps = int(total_steps * config['warmup_ratio'])

# Initialize the learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=config['min_learning_rate'])

def get_warmup_lr(step, warmup_steps, initial_lr):
    return initial_lr * (step + 1) / warmup_steps

# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Prepare all training components with accelerator
llava_model, optimizer, dataloader, scheduler = accelerator.prepare(
    llava_model, optimizer, dataloader, scheduler
)

# Training loop
llava_model.train()
global_step = 0
for epoch in range(config['n_epochs']):
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=min(config['n_batches'], len(dataloader)), desc=f"Epoch {epoch+1}/{config['n_epochs']}")
    
    for step, batch in progress_bar:

        if step >= config['n_batches']:
            break

        if global_step < warmup_steps:
            warmup_lr = get_warmup_lr(
                global_step, 
                warmup_steps, 
                config['initial_learning_rate']
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            scheduler.step()
        
        # Use Accelerate's accumulate context manager
        with accelerator.accumulate(llava_model):

            current_lr = optimizer.param_groups[0]['lr']

            # Move tensors to device
            image = batch["image_tensor"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            assistant_positions = batch["assistant_positions"]

            # Forward pass
            outputs = llava_model(image, input_ids, attention_mask)
            vision_token_offset = outputs.shape[1] - input_ids.shape[1]

            # Calculate loss for each assistant response
            batch_loss = 0        
            for b in range(outputs.shape[0]):
                for start_pos, end_pos in assistant_positions[b]:
                    response_logits = outputs[b, vision_token_offset+start_pos-1:vision_token_offset+end_pos-1, :]
                    response_targets = input_ids[b, start_pos:end_pos]
                    loss = criterion(
                        response_logits.contiguous().view(-1, outputs.size(-1)),
                        response_targets.contiguous().view(-1)
                    )
                    batch_loss += loss / len(assistant_positions[b])

            # Average loss over batch
            batch_loss = batch_loss / outputs.shape[0]
            
            # Backward pass
            accelerator.backward(batch_loss)
            
            # Step optimizer (Accelerate handles this automatically)
            optimizer.step()
            optimizer.zero_grad()
            
            # Log metrics
            step_logs = {
                "batch_loss": batch_loss.item(),
                "learning_rate": current_lr,
                "global_step": global_step
            }
            if accelerator.is_main_process:
                wandb.log(step_logs)        
            progress_bar.set_postfix({"Loss": f"{batch_loss.item():.4f}", "LR": f"{current_lr:.6f}"})

        global_step += 1

        # Save the projector weights every 'save_every' steps
        if global_step % config['save_every'] == 0:
            llava_model.save_projector_weights(PROJECTOR_WEIGHTS_PATH + ".pth")
            if accelerator.is_main_process:
                wandb.save(PROJECTOR_WEIGHTS_PATH + ".pth")    

llava_model.save_projector_weights(PROJECTOR_WEIGHTS_PATH + ".pth")

# End the wandb run
if accelerator.is_main_process:
    wandb.finish()

# %%
llava_model.eval()  # Set the model to evaluation mode

# Select a random example from the dataset
random_idx = torch.randint(0, len(dataset), (1,)).item()
sample = dataset[random_idx]

# Prepare the input
image_tensor = sample["image_tensor"].to(device)

# input_ids until first "108" token
input_ids_ = sample["input_ids"].unsqueeze(0).to(device)
attention_mask_ = sample["attention_mask"].unsqueeze(0).to(device)

cutoff = input_ids_[0].tolist().index(2516) + 1 
input_ids = input_ids_[:, :cutoff]
attention_mask = attention_mask_[:, :cutoff]

# Autoregressive generation
max_new_tokens = 50  # Maximum number of tokens to generate
temperature = 0.001 # Adjust this value to control randomness (e.g., 0.5 to 1.5)
generated_tokens = []
current_input_ids = input_ids.clone()
current_attention_mask = attention_mask.clone()

for _ in range(max_new_tokens):
    with torch.no_grad():
        outputs = llava_model(image_tensor, current_input_ids, current_attention_mask)
    
    next_token_logits = outputs[:, -1, :] / temperature
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze()
    
    generated_tokens.append(next_token.item())
    if next_token.item() == 107:
        break
    
    current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=device)], dim=1)

# Decode the input, true output, and generated output
input_text = tokenizer.decode(input_ids[0])
cutoff_output = cutoff + input_ids_[0, cutoff:].tolist().index(107)
true_output = tokenizer.decode(input_ids_[0,cutoff:cutoff_output])
generated_output = tokenizer.decode(generated_tokens)

# Print the results
print("Input:")
print(input_text)
print("\nTrue Output:")
print(true_output)
print("\nGenerated Output (Temperature =", temperature, "):")
print(generated_output)

plt.figure(figsize=(5, 5))
plt.imshow(dataset.get_image(sample["image_idx"]))
plt.axis('off')
plt.show()

# %%