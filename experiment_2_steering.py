import torch
import transformer_lens
from transformer_lens import HookedTransformer

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

# 1. Load the model
print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# 2. Define the prompts
pos_prompt = "The food was wonderful, amazing, and fantastic."
neg_prompt = "The food was terrible, awful, and disgusting."
neutral_prompt = "I went to the restaurant and the food was"

# We will steer the residual stream at a middle layer
steering_layer = 6

# 3. Calculate the steering vector
print("Calculating steering vector...")
pos_logits, pos_cache = model.run_with_cache(pos_prompt)
neg_logits, neg_cache = model.run_with_cache(neg_prompt)

# Extract the residual stream at the target layer for the last token
pos_res = pos_cache[f"blocks.{steering_layer}.hook_resid_post"][0, -1, :]
neg_res = neg_cache[f"blocks.{steering_layer}.hook_resid_post"][0, -1, :]

# The direction vector points from negative to positive sentiment
steering_vector = pos_res - neg_res

# Normalize the vector
steering_vector = steering_vector / torch.norm(steering_vector)

# 4. Generate Baseline (Unsteered)
print("\n--- Baseline Generation ---")
torch.manual_seed(42) # For reproducibility
baseline_gen = model.generate(neutral_prompt, max_new_tokens=20, temperature=0.7)
print(f"Baseline: {baseline_gen}")

# 5. Generate Steered (Positive)
print("\n--- Steered Generation (Positive) ---")
# Define the hook function to inject the vector
def positive_steering_hook(value, hook, multiplier=5.0):
    # value shape: [batch, pos, d_model]
    # Add the steering vector to all positions
    value[:, :, :] += steering_vector * multiplier
    return value

# Add the hook and generate
torch.manual_seed(42) # Same seed to isolate the steering effect
with model.hooks([(f"blocks.{steering_layer}.hook_resid_post", positive_steering_hook)]):
    pos_steered_gen = model.generate(neutral_prompt, max_new_tokens=20, temperature=0.7)
print(f"Steered (+ Positive): {pos_steered_gen}")

# 6. Generate Steered (Negative)
print("\n--- Steered Generation (Negative) ---")
# To steer negatively, we subtract the vector (or use a negative multiplier)
def negative_steering_hook(value, hook, multiplier=-5.0):
    value[:, :, :] += steering_vector * multiplier
    return value

torch.manual_seed(42)
with model.hooks([(f"blocks.{steering_layer}.hook_resid_post", negative_steering_hook)]):
    neg_steered_gen = model.generate(neutral_prompt, max_new_tokens=20, temperature=0.7)
print(f"Steered (- Negative): {neg_steered_gen}")

print("\n--- Conclusion ---")
print("By simply adding a 'sentiment' vector to the model's internal residual stream at layer 6,")
print("we successfully steered the model's generated text without modifying its prompt or parameters.")
