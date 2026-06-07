import torch
import transformer_lens
from transformer_lens import HookedTransformer
import plotly.graph_objects as go
import numpy as np

device = "cpu"

print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# 1. Calculate the steering vector
pos_prompt = "The food was wonderful, amazing, and fantastic."
neg_prompt = "The food was terrible, awful, and disgusting."
neutral_prompt = "I went to the restaurant and the food was"

steering_layer = 6

pos_logits, pos_cache = model.run_with_cache(pos_prompt)
neg_logits, neg_cache = model.run_with_cache(neg_prompt)

pos_res = pos_cache[f"blocks.{steering_layer}.hook_resid_post"][0, -1, :]
neg_res = neg_cache[f"blocks.{steering_layer}.hook_resid_post"][0, -1, :]

steering_vector = pos_res - neg_res
steering_vector = steering_vector / torch.norm(steering_vector)

# 2. Define tokens to track
# GPT-2 tokenizes words with leading spaces
positive_words = [" amazing", " wonderful", " great", " fantastic", " delicious", " excellent"]
negative_words = [" terrible", " awful", " disgusting", " bad", " horrible", " gross"]

positive_tokens = [model.to_single_token(w) for w in positive_words]
negative_tokens = [model.to_single_token(w) for w in negative_words]

# 3. Sweep over multipliers and record probabilities
multipliers = np.linspace(-15, 15, 31)
pos_probs_list = []
neg_probs_list = []

print("Running steering sweeps...")
for mult in multipliers:
    def steering_hook(value, hook):
        value[:, :, :] += steering_vector * mult
        return value

    with model.hooks([(f"blocks.{steering_layer}.hook_resid_post", steering_hook)]):
        logits = model(neutral_prompt)
        
    next_token_logits = logits[0, -1, :]
    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
    
    # Sum the probabilities of our target word sets
    pos_prob = sum([probs[t].item() for t in positive_tokens])
    neg_prob = sum([probs[t].item() for t in negative_tokens])
    
    pos_probs_list.append(pos_prob)
    neg_probs_list.append(neg_prob)

# 4. Generate Plot
print("Generating plot...")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=multipliers, y=pos_probs_list,
    mode='lines+markers',
    name='Positive Words (e.g. "amazing", "delicious")',
    line=dict(color='green', width=3),
    marker=dict(size=8)
))

fig.add_trace(go.Scatter(
    x=multipliers, y=neg_probs_list,
    mode='lines+markers',
    name='Negative Words (e.g. "terrible", "disgusting")',
    line=dict(color='red', width=3),
    marker=dict(size=8)
))

fig.update_layout(
    title="Effect of Feature Steering (Activation Addition) on GPT-2 Generation",
    xaxis_title="Steering Multiplier (Strength of injected Sentiment Vector)",
    yaxis_title="Probability of generating specific word class",
    template="plotly_dark",
    font=dict(size=14),
    hovermode="x unified",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0,0,0,0.5)"
    )
)

# Save to file
output_file = "feature_steering_plot.png"
fig.write_image(output_file, width=1000, height=600, scale=2)
print(f"Plot saved successfully to {output_file}")
