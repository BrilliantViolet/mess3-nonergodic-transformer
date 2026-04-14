# -*- coding: utf-8 -*-

import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)

"""Colab notebooks execute code on Google's cloud servers, meaning you can leverage the power of Google hardware, including [GPUs and TPUs](#using-accelerated-hardware), regardless of the power of your machine. All you need is a browser.

For example, if you find yourself waiting for **pandas** code to finish running and want to go faster, you can switch to a GPU Runtime and use libraries like [RAPIDS cuDF](https://rapids.ai/cudf-pandas) that provide zero-code-change acceleration.

To learn more about accelerating pandas on Colab, see the [10 minute guide](https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cudf_pandas_colab_demo.ipynb) or
 [US stock market data analysis demo](https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cudf_pandas_stocks_demo.ipynb).

<div class="markdown-google-sans">

## Machine learning
</div>

With Colab you can import an image dataset, train an image classifier on it, and evaluate the model, all in just [a few lines of code](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb).

Colab is used extensively in the machine learning community with applications including:
- Getting started with TensorFlow
- Developing and training neural networks
- Experimenting with TPUs
- Disseminating AI research
- Creating tutorials

To see sample Colab notebooks that demonstrate machine learning applications, see the [machine learning examples](#machine-learning-examples) below.

<div class="markdown-google-sans">

## More Resources

### Working with Notebooks in Colab

</div>

- [Overview of Colab](/notebooks/basic_features_overview.ipynb)
- [Guide to Markdown](/notebooks/markdown_guide.ipynb)
- [Importing libraries and installing dependencies](/notebooks/snippets/importing_libraries.ipynb)
- [Saving and loading notebooks in GitHub](https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb)
- [Interactive forms](/notebooks/forms.ipynb)
- [Interactive widgets](/notebooks/widgets.ipynb)

<div class="markdown-google-sans">

<a name="working-with-data"></a>
### Working with Data
</div>

- [Loading data: Drive, Sheets, and Google Cloud Storage](/notebooks/io.ipynb)
- [Charts: visualizing data](/notebooks/charts.ipynb)
- [Getting started with BigQuery](/notebooks/bigquery.ipynb)

<div class="markdown-google-sans">

### Machine Learning

<div>

These are a few of the notebooks related to Machine Learning, including Google's online Machine Learning course. See the [full course website](https://developers.google.com/machine-learning/crash-course/) for more.
- [Intro to Pandas DataFrame](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/pandas_dataframe_ultraquick_tutorial.ipynb)
- [Intro to RAPIDS cuDF to accelerate pandas](https://nvda.ws/rapids-cudf)
- [Getting Started with cuML's accelerator mode](https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cuml_sklearn_colab_demo.ipynb)

<div class="markdown-google-sans">

<a name="using-accelerated-hardware"></a>
### Using Accelerated Hardware
</div>

- [Train a CNN to classify handwritten digits on the MNIST dataset using Flax NNX API](https://colab.research.google.com/github/google/flax/blob/main/docs_nnx/mnist_tutorial.ipynb)
- [Train a Vision Transformer (ViT) for image classification with JAX](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_Vision_transformer.ipynb)
- [Text classification with a transformer language model using JAX](https://colab.research.google.com/github/jax-ml/jax-ai-stack/blob/main/docs/source/JAX_transformer_text_classification.ipynb)

<div class="markdown-google-sans">

<a name="machine-learning-examples"></a>

### Featured examples

</div>

- [Train a miniGPT language model with JAX AI Stack](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html)
- [LoRA/QLoRA finetuning for LLM using Tunix](https://github.com/google/tunix/blob/main/examples/qlora_gemma.ipynb)
- [Parameter-efficient fine-tuning of Gemma with LoRA and QLoRA](https://keras.io/examples/keras_recipes/parameter_efficient_finetuning_of_gemma_with_lora_and_qlora/)
- [Loading Hugging Face Transformers Checkpoints](https://keras.io/keras_hub/guides/hugging_face_keras_integration/)
- [8-bit Integer Quantization in Keras](https://keras.io/guides/int8_quantization_in_keras/)
- [Float8 training and inference with a simple Transformer model](https://keras.io/examples/keras_recipes/float8_training_and_inference_with_transformer/)
- [Pretraining a Transformer from scratch with KerasHub](https://keras.io/keras_hub/guides/transformer_pretraining/)
- [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet/)
- [Image classification from scratch using Keras 3](https://keras.io/examples/vision/image_classification_from_scratch/)
- [Image Classification with KerasHub](https://keras.io/keras_hub/guides/classification_with_keras_hub/)

!pip install torch numpy matplotlib scikit-learn
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Libraries loaded successfully")

!pip install torch numpy matplotlib scikit-learn

# Mess3 data generation
# Parameters for two different ergodic components
def mess3_transition_matrix(x, a):
    """
    Generate Mess3 transition matrix.
    x: self-transition probability
    a: asymmetry parameter
    """
    T = np.array([
        [x, (1-x)*a, (1-x)*(1-a)],
        [(1-x)*(1-a), x, (1-x)*a],
        [(1-x)*a, (1-x)*(1-a), x]
    ])
    return T

def generate_mess3_sequence(T, length, start_state=None):
    """Generate token sequence from Mess3 HMM."""
    n_states = T.shape[0]
    if start_state is None:
        state = np.random.randint(n_states)
    else:
        state = start_state

    sequence = []
    for _ in range(length):
        token = state  # token = state (simplified)
        sequence.append(token)
        state = np.random.choice(n_states, p=T[state])

    return np.array(sequence)

# Component A: slow mixing (high inertia)
T_A = mess3_transition_matrix(x=0.7, a=0.6)
# Component B: fast mixing (low inertia)
T_B = mess3_transition_matrix(x=0.3, a=0.6)

print("Component A transition matrix:")
print(T_A.round(3))
print("\nComponent B transition matrix:")
print(T_B.round(3))
print("\nData generation ready")# Mess3 data generation
# Parameters for two different ergodic components
def mess3_transition_matrix(x, a):
    """
    Generate Mess3 transition matrix.
    x: self-transition probability
    a: asymmetry parameter
    """
    T = np.array([
        [x, (1-x)*a, (1-x)*(1-a)],
        [(1-x)*(1-a), x, (1-x)*a],
        [(1-x)*a, (1-x)*(1-a), x]
    ])
    return T

def generate_mess3_sequence(T, length, start_state=None):
    """Generate token sequence from Mess3 HMM."""
    n_states = T.shape[0]
    if start_state is None:
        state = np.random.randint(n_states)
    else:
        state = start_state

    sequence = []
    for _ in range(length):
        token = state  # token = state (simplified)
        sequence.append(token)
        state = np.random.choice(n_states, p=T[state])

    return np.array(sequence)

# Component A: slow mixing (high inertia)
T_A = mess3_transition_matrix(x=0.7, a=0.6)
# Component B: fast mixing (low inertia)
T_B = mess3_transition_matrix(x=0.3, a=0.6)

print("Component A transition matrix:")
print(T_A.round(3))
print("\nComponent B transition matrix:")
print(T_B.round(3))
print("\nData generation ready")# Mess3 data generation
# Parameters for two different ergodic components
def mess3_transition_matrix(x, a):
    """
    Generate Mess3 transition matrix.
    x: self-transition probability
    a: asymmetry parameter
    """
    T = np.array([
        [x, (1-x)*a, (1-x)*(1-a)],
        [(1-x)*(1-a), x, (1-x)*a],
        [(1-x)*a, (1-x)*(1-a), x]
    ])
    return T

def generate_mess3_sequence(T, length, start_state=None):
    """Generate token sequence from Mess3 HMM."""
    n_states = T.shape[0]
    if start_state is None:
        state = np.random.randint(n_states)
    else:
        state = start_state

    sequence = []
    for _ in range(length):
        token = state  # token = state (simplified)
        sequence.append(token)
        state = np.random.choice(n_states, p=T[state])

    return np.array(sequence)

# Component A: slow mixing (high inertia)
T_A = mess3_transition_matrix(x=0.7, a=0.6)
# Component B: fast mixing (low inertia)
T_B = mess3_transition_matrix(x=0.3, a=0.6)

print("Component A transition matrix:")
print(T_A.round(3))
print("\nComponent B transition matrix:")
print(T_B.round(3))
print("\nData generation ready")

# Generate training data (practice block + main session)
# Practice block: warm-up sequences
VOCAB_SIZE = 3
SEQ_LEN = 16
N_WARMUP = 200      # practice block
N_TRAIN = 2000      # main session
N_TEST = 200        # test session

def generate_dataset(T, n_sequences, seq_len):
    data = []
    for _ in range(n_sequences):
        seq = generate_mess3_sequence(T, seq_len + 1)
        data.append(seq)
    data = np.array(data)
    inputs = torch.tensor(data[:, :-1], dtype=torch.long)
    targets = torch.tensor(data[:, 1:], dtype=torch.long)
    return inputs, targets

# Practice block (component A only)
warmup_inputs, warmup_targets = generate_dataset(T_A, N_WARMUP, SEQ_LEN)

# Main session (mix of A and B)
train_inputs_A, train_targets_A = generate_dataset(T_A, N_TRAIN // 2, SEQ_LEN)
train_inputs_B, train_targets_B = generate_dataset(T_B, N_TRAIN // 2, SEQ_LEN)
train_inputs = torch.cat([train_inputs_A, train_inputs_B], dim=0)
train_targets = torch.cat([train_targets_A, train_targets_B], dim=0)

# Shuffle
idx = torch.randperm(train_inputs.shape[0])
train_inputs = train_inputs[idx]
train_targets = train_targets[idx]

print(f"Practice block: {warmup_inputs.shape}")
print(f"Main session: {train_inputs.shape}")
print("Dataset ready")# Generate training data (practice block + main session)
# Practice block: warm-up sequences
VOCAB_SIZE = 3
SEQ_LEN = 16
N_WARMUP = 200      # practice block
N_TRAIN = 2000      # main session
N_TEST = 200        # test session

def generate_dataset(T, n_sequences, seq_len):
    data = []
    for _ in range(n_sequences):
        seq = generate_mess3_sequence(T, seq_len + 1)
        data.append(seq)
    data = np.array(data)
    inputs = torch.tensor(data[:, :-1], dtype=torch.long)
    targets = torch.tensor(data[:, 1:], dtype=torch.long)
    return inputs, targets

# Practice block (component A only)
warmup_inputs, warmup_targets = generate_dataset(T_A, N_WARMUP, SEQ_LEN)

# Main session (mix of A and B)
train_inputs_A, train_targets_A = generate_dataset(T_A, N_TRAIN // 2, SEQ_LEN)
train_inputs_B, train_targets_B = generate_dataset(T_B, N_TRAIN // 2, SEQ_LEN)
train_inputs = torch.cat([train_inputs_A, train_inputs_B], dim=0)
train_targets = torch.cat([train_targets_A, train_targets_B], dim=0)

# Shuffle
idx = torch.randperm(train_inputs.shape[0])
train_inputs = train_inputs[idx]
train_targets = train_targets[idx]

print(f"Practice block: {warmup_inputs.shape}")
print(f"Main session: {train_inputs.shape}")
print("Dataset ready")# Generate training data (practice block + main session)
# Practice block: warm-up sequences
VOCAB_SIZE = 3
SEQ_LEN = 16
N_WARMUP = 200      # practice block
N_TRAIN = 2000      # main session
N_TEST = 200        # test session

def generate_dataset(T, n_sequences, seq_len):
    data = []
    for _ in range(n_sequences):
        seq = generate_mess3_sequence(T, seq_len + 1)
        data.append(seq)
    data = np.array(data)
    inputs = torch.tensor(data[:, :-1], dtype=torch.long)
    targets = torch.tensor(data[:, 1:], dtype=torch.long)
    return inputs, targets

# Practice block (component A only)
warmup_inputs, warmup_targets = generate_dataset(T_A, N_WARMUP, SEQ_LEN)

# Main session (mix of A and B)
train_inputs_A, train_targets_A = generate_dataset(T_A, N_TRAIN // 2, SEQ_LEN)
train_inputs_B, train_targets_B = generate_dataset(T_B, N_TRAIN // 2, SEQ_LEN)
train_inputs = torch.cat([train_inputs_A, train_inputs_B], dim=0)
train_targets = torch.cat([train_targets_A, train_targets_B], dim=0)

# Shuffle
idx = torch.randperm(train_inputs.shape[0])
train_inputs = train_inputs[idx]
train_targets = train_targets[idx]

print(f"Practice block: {warmup_inputs.shape}")
print(f"Main session: {train_inputs.shape}")
print("Dataset ready")

# Generate training data (practice block + main session)
VOCAB_SIZE = 3
SEQ_LEN = 16
N_WARMUP = 200
N_TRAIN = 2000
N_TEST = 200

def generate_dataset(T, n_sequences, seq_len):
    data = []
    for _ in range(n_sequences):
        seq = generate_mess3_sequence(T, seq_len + 1)
        data.append(seq)
    data = np.array(data)
    inputs = torch.tensor(data[:, :-1], dtype=torch.long)
    targets = torch.tensor(data[:, 1:], dtype=torch.long)
    return inputs, targets

warmup_inputs, warmup_targets = generate_dataset(T_A, N_WARMUP, SEQ_LEN)
train_inputs_A, train_targets_A = generate_dataset(T_A, N_TRAIN // 2, SEQ_LEN)
train_inputs_B, train_targets_B = generate_dataset(T_B, N_TRAIN // 2, SEQ_LEN)
train_inputs = torch.cat([train_inputs_A, train_inputs_B], dim=0)
train_targets = torch.cat([train_targets_A, train_targets_B], dim=0)
idx = torch.randperm(train_inputs.shape[0])
train_inputs = train_inputs[idx]
train_targets = train_targets[idx]

print(f"Practice block: {warmup_inputs.shape}")
print(f"Main session: {train_inputs.shape}")
print("Dataset ready")

# Define small Transformer model
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2, seq_len=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=128, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x, return_residual=False):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        residual = self.transformer(h, mask=mask, is_causal=True)
        logits = self.output(residual)
        if return_residual:
            return logits, residual
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SmallTransformer(VOCAB_SIZE).to(device)
print(f"Device: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Model ready")

# Training (practice block + main session)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def train_epoch(inputs, targets, n_epochs, desc="Training"):
    model.train()
    losses = []
    for epoch in range(n_epochs):
        inputs_dev = inputs.to(device)
        targets_dev = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs_dev)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets_dev.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"{desc} | Epoch {epoch+1} | Loss: {loss.item():.4f}")
    return losses

# Practice block (warmup)
print("=== Practice Block (Warmup) ===")
warmup_losses = train_epoch(warmup_inputs, warmup_targets, n_epochs=20, desc="Warmup")

# Main session
print("\n=== Main Session ===")
train_losses = train_epoch(train_inputs, train_targets, n_epochs=100, desc="Train")

print("\nTraining complete")

# Test session - generate switching sequences
# No-cue condition: abrupt switch from A to B
# Cue condition: special token inserted before switch

def generate_switch_sequence(T_A, T_B, seq_len=16, switch_point=8, cue=False):
    """Generate sequence that switches from component A to B."""
    seq_A = generate_mess3_sequence(T_A, switch_point)
    seq_B = generate_mess3_sequence(T_B, seq_len - switch_point)

    if cue:
        # Insert cue token (use token 3 as cue - outside normal vocab)
        seq = np.concatenate([seq_A[:-1], [2], seq_B])[:seq_len]
    else:
        seq = np.concatenate([seq_A, seq_B])[:seq_len]

    return seq, switch_point

# Generate test sequences
N_SWITCH_SEQS = 100
switch_point = 8

# No-cue condition
nocue_seqs = []
for _ in range(N_SWITCH_SEQS):
    seq, sp = generate_switch_sequence(T_A, T_B, SEQ_LEN, switch_point, cue=False)
    nocue_seqs.append(seq)
nocue_seqs = torch.tensor(np.array(nocue_seqs), dtype=torch.long)

# Cue condition
cue_seqs = []
for _ in range(N_SWITCH_SEQS):
    seq, sp = generate_switch_sequence(T_A, T_B, SEQ_LEN, switch_point, cue=True)
    cue_seqs.append(seq)
cue_seqs = torch.tensor(np.array(cue_seqs), dtype=torch.long)

print(f"No-cue sequences: {nocue_seqs.shape}")
print(f"Cue sequences: {cue_seqs.shape}")
print(f"Switch point: position {switch_point}")
print("Test sequences ready")

# Measure switching cost: RT (reconvergence speed) and Accuracy
def measure_switching_cost(model, sequences, switch_point, T_before, T_after):
    model.eval()
    all_accuracies = []
    all_losses = []

    with torch.no_grad():
        inputs = sequences[:, :-1].to(device)
        targets = sequences[:, 1:].to(device)
        logits, residuals = model(inputs, return_residual=True)

        # Per-position accuracy and loss
        for pos in range(inputs.shape[1]):
            pos_logits = logits[:, pos, :]
            pos_targets = targets[:, pos]

            # Accuracy
            preds = pos_logits.argmax(dim=-1)
            acc = (preds == pos_targets).float().mean().item()
            all_accuracies.append(acc)

            # Loss
            loss = criterion(pos_logits, pos_targets).item()
            all_losses.append(loss)

    return np.array(all_accuracies), np.array(all_losses)

# No-cue condition
nocue_acc, nocue_loss = measure_switching_cost(
    model, nocue_seqs, switch_point, T_A, T_B
)

# Cue condition
cue_acc, cue_loss = measure_switching_cost(
    model, cue_seqs, switch_point, T_A, T_B
)

print("Switching cost measurement complete")
print(f"No-cue accuracy around switch: {nocue_acc[switch_point-2:switch_point+4].round(3)}")
print(f"Cue accuracy around switch: {cue_acc[switch_point-2:switch_point+4].round(3)}")

# Visualization - Switching Cost (RT and Accuracy)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

positions = np.arange(len(nocue_acc))

# Accuracy plot
axes[0].plot(positions, nocue_acc, 'b-o', label='No-cue', markersize=4)
axes[0].plot(positions, cue_acc, 'r-o', label='Cue', markersize=4)
axes[0].axvline(x=switch_point, color='k', linestyle='--', label='Switch point')
axes[0].set_xlabel('Token position')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy: Switching Cost by Condition')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot (RT analog)
axes[1].plot(positions, nocue_loss, 'b-o', label='No-cue', markersize=4)
axes[1].plot(positions, cue_loss, 'r-o', label='Cue', markersize=4)
axes[1].axvline(x=switch_point, color='k', linestyle='--', label='Switch point')
axes[1].set_xlabel('Token position')
axes[1].set_ylabel('Loss (RT analog)')
axes[1].set_title('Loss: Switching Cost by Condition')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Transformer Task Switching Cost\n(Non-ergodic Mess3 Components)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('switching_cost.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as switching_cost.png")

# Summary of switching cost results
import json

# RT analog: tokens needed to reconverge after switch
def reconvergence_speed(acc_array, switch_point, threshold=0.5):
    """How many tokens after switch until accuracy recovers above threshold."""
    post_switch = acc_array[switch_point:]
    for i, acc in enumerate(post_switch):
        if acc >= threshold:
            return i
    return len(post_switch)

nocue_rt = reconvergence_speed(nocue_acc, switch_point)
cue_rt = reconvergence_speed(cue_acc, switch_point)

# Accuracy drop at switch point
nocue_acc_drop = nocue_acc[switch_point-1] - nocue_acc[switch_point]
cue_acc_drop = cue_acc[switch_point-1] - cue_acc[switch_point]

print("=== Switching Cost Results ===")
print(f"\nRT analog (tokens to reconverge):")
print(f"  No-cue condition: {nocue_rt} tokens")
print(f"  Cue condition:    {cue_rt} tokens")
print(f"  Cue benefit (RT): {nocue_rt - cue_rt} tokens faster")

print(f"\nAccuracy drop at switch point:")
print(f"  No-cue condition: {nocue_acc_drop:.3f}")
print(f"  Cue condition:    {cue_acc_drop:.3f}")

# Save results
results = {
    "nocue_accuracy": nocue_acc.tolist(),
    "cue_accuracy": cue_acc.tolist(),
    "nocue_loss": nocue_loss.tolist(),
    "cue_loss": cue_loss.tolist(),
    "switch_point": switch_point,
    "nocue_rt": nocue_rt,
    "cue_rt": cue_rt,
    "nocue_acc_drop": float(nocue_acc_drop),
    "cue_acc_drop": float(cue_acc_drop)
}

with open('switching_cost_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to switching_cost_results.json")

# Visualization - Switching Cost (RT and Accuracy)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

positions = np.arange(len(nocue_acc))

# Accuracy plot
axes[0].plot(positions, nocue_acc, 'b-o', label='No-cue', markersize=4)
axes[0].plot(positions, cue_acc, 'r-o', label='Cue', markersize=4)
axes[0].axvline(x=switch_point, color='k', linestyle='--', label='Switch point')
axes[0].set_xlabel('Token position')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy: Switching Cost by Condition')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot (RT analog)
axes[1].plot(positions, nocue_loss, 'b-o', label='No-cue', markersize=4)
axes[1].plot(positions, cue_loss, 'r-o', label='Cue', markersize=4)
axes[1].axvline(x=switch_point, color='k', linestyle='--', label='Switch point')
axes[1].set_xlabel('Token position')
axes[1].set_ylabel('Loss (RT analog)')
axes[1].set_title('Loss: Switching Cost by Condition')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Transformer Task Switching Cost\n(Non-ergodic Mess3 Components)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('switching_cost.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved as switching_cost.png")

