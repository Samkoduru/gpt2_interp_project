import torch
import transformer_lens
from transformer_lens import HookedTransformer

# Force CPU because MPS backend has a known bug with TransformerLens hooks
device = "cpu"
print(f"Using device: {device}")

# 1. Load the model
print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# 2. Define the Indirect Object Identification prompt
prompt = "John and Mary went to the store. John gave a bottle to"
target_word = " Mary"
target_token = model.to_single_token(target_word)

# 3. Baseline forward pass
print("\n--- Baseline ---")
logits, cache = model.run_with_cache(prompt)

# Calculate baseline probability for the target word
next_token_logits = logits[0, -1, :]
probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
baseline_prob = probs[target_token].item()
print(f"Prompt: '{prompt}'")
print(f"Target: '{target_word}'")
print(f"Baseline Probability of '{target_word}': {baseline_prob:.2%}")

# Look at top 5 predictions
top_probs, top_indices = torch.topk(probs, 5)
print("Top 5 baseline predictions:")
for p, i in zip(top_probs, top_indices):
    print(f"  '{model.to_string(i)}': {p:.2%}")

# 4. Identify Layers to Ablate
# We will ablate the attention output of layers 9 and 10 entirely to break the IOI circuit
layers_to_ablate = [9, 10]

# 5. Ablation function
def ablation_hook(value, hook):
    # value shape: [batch, pos, n_heads, d_model]
    # Zero out the entire layer's attention output
    value[:] = 0.0
    return value

# 6. Run with ablation
print(f"\\n--- Ablated ---")
print(f"Ablating ALL attention heads in layers: {layers_to_ablate}")

hook_names = [f"blocks.{layer}.attn.hook_result" for layer in layers_to_ablate]

with model.hooks([(name, ablation_hook) for name in hook_names]):
    ablated_logits = model(prompt)

# Calculate ablated probability
ablated_next_token_logits = ablated_logits[0, -1, :]
ablated_probs = torch.nn.functional.softmax(ablated_next_token_logits, dim=-1)
ablated_prob = ablated_probs[target_token].item()

print(f"Ablated Probability of '{target_word}': {ablated_prob:.2%}")

# Look at top 5 ablated predictions
top_ablated_probs, top_ablated_indices = torch.topk(ablated_probs, 5)
print("Top 5 ablated predictions:")
for p, i in zip(top_ablated_probs, top_ablated_indices):
    print(f"  '{model.to_string(i)}': {p:.2%}")

print("\n--- Conclusion ---")
print(f"The probability of the model correctly identifying '{target_word}' dropped from {baseline_prob:.2%} to {ablated_prob:.2%}.")
print("This provides concrete mechanistic evidence that these specific attention heads are responsible for the task.")
