import torch
import transformer_lens
from transformer_lens import HookedTransformer
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

device = "cpu"
print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

prompt = "John and Mary went to the store. John gave a bottle to"
target_word = " Mary"
target_token = model.to_single_token(target_word)

print("Running model with cache...")
logits, cache = model.run_with_cache(prompt)
tokens = model.to_str_tokens(prompt)

# ==========================================
# Plot 1: Logit Lens
# ==========================================
print("Generating Logit Lens Plot...")
# Logit Lens looks at the residual stream at each layer and applies the final unembedding matrix
# to see what the model is "thinking" at that exact layer.

layer_probs = []
layers = list(range(model.cfg.n_layers))

for layer in layers:
    # Get the residual stream at this layer for the final token
    resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
    
    # Apply LayerNorm and Unembed to get logits
    scaled_resid = model.ln_final(resid)
    layer_logits = model.unembed(scaled_resid)
    
    # Get probability of target token
    probs = torch.nn.functional.softmax(layer_logits, dim=-1)
    layer_probs.append(probs[target_token].item())

fig_logit = go.Figure()
fig_logit.add_trace(go.Scatter(
    x=layers, y=layer_probs,
    mode='lines+markers',
    name=f'Probability of "{target_word}"',
    line=dict(color='cyan', width=4),
    marker=dict(size=10, color='white')
))

fig_logit.update_layout(
    title="Logit Lens: When does GPT-2 'decide' on the answer?",
    xaxis_title="Transformer Layer",
    yaxis_title=f"Probability of predicting '{target_word}'",
    template="plotly_dark",
    font=dict(size=14)
)
fig_logit.write_image("logit_lens_plot.png", width=1000, height=600, scale=2)
print("Saved logit_lens_plot.png")

# ==========================================
# Plot 2: Attention Pattern Heatmap (Name Mover Head)
# ==========================================
print("Generating Attention Heatmap...")
# Layer 9, Head 9 is a known "Name Mover Head" for the IOI task.
layer_idx = 9
head_idx = 9

# Get the attention pattern for this specific layer: shape [batch, n_heads, query_pos, key_pos]
attn_pattern = cache[f"blocks.{layer_idx}.attn.hook_pattern"][0, head_idx, :, :]

# Convert to numpy for plotly
attn_matrix = attn_pattern.detach().cpu().numpy()

# We want to show the attention from each query token to each key token.
fig_attn = px.imshow(
    attn_matrix,
    x=tokens,
    y=tokens,
    labels=dict(x="Key (Word being looked at)", y="Query (Current Word)"),
    title=f"Attention Pattern of Name Mover Head (Layer {layer_idx}, Head {head_idx})",
    color_continuous_scale="viridis"
)

# Emphasize the specific cell where the final token " to" looks at " Mary"
# Find index of " to" and " Mary"
to_idx = len(tokens) - 1
mary_idx = tokens.index(" Mary")

fig_attn.add_shape(
    type="rect",
    x0=mary_idx - 0.5, y0=to_idx - 0.5,
    x1=mary_idx + 0.5, y1=to_idx + 0.5,
    line=dict(color="red", width=3)
)

fig_attn.update_layout(
    template="plotly_dark",
    font=dict(size=14),
    xaxis_tickangle=-45
)

fig_attn.write_image("attention_heatmap.png", width=1000, height=800, scale=2)
print("Saved attention_heatmap.png")
