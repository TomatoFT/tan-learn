# tan-gpt

A small GPT built from the pieces underneath it. Each module is a lesson with a `main` that checks itself. Run them from this directory:

```bash
pip install -r requirements.txt
python -m foundations.gradient_descent
python -m foundations.train
python -m foundations.generate
```

`train` writes `checkpoints/tiny-gpt.pt`. `generate` reads that file. The default corpus is a short repeated passage, and the default run is 300 steps, so a tiny character model can show the loss falling on a CPU in well under a minute.

## foundations

| Module | What it covers |
| --- | --- |
| `gradient_descent.py` | Gradient descent optimizer |
| `activations.py` | Activation functions |
| `softmax.py` | Softmax function |
| `loss.py` | Cross-entropy loss |
| `linear_regression.py` | Linear regression forward pass |
| `linear_regression_training.py` | Linear regression training |
| `neuron.py` | Single neuron forward pass |
| `backprop.py` | Backpropagation |
| `multi_layer_backprop.py` | Multi-layer backpropagation |
| `mlp.py` | MLP from scratch |
| `weight_init.py` | Weight initialization |
| `pytorch_basics.py` | PyTorch basics |
| `training_loop.py` | Training loop |
| `training_diagnostics.py` | Training diagnostics |
| `dead_relu_detector.py` | Dead ReLU detector |
| `digit_classifier.py` | Digit classifier |
| `sentiment.py` | Sentiment analysis |
| `train.py` | GPT training loop |
| `generate.py` | Text generation |

## data

| Module | What it covers |
| --- | --- |
| `nlp_preprocessing.py` | NLP preprocessing |
| `tokenizer.py` | BPE tokenizer |
| `vocab.py` | Character-level encode/decode |
| `tokenizer_utils.py` | Tokenization edge cases |
| `loader.py` | Batched training data |
| `dataset.py` | GPT dataset preparation |

## model

| Module | What it covers |
| --- | --- |
| `normalization.py` | Layer normalization |
| `batch_normalization.py` | Batch normalization |
| `rms_normalization.py` | RMS normalization |
| `embeddings.py` | Word embeddings |
| `positional_encoding.py` | Positional encoding |
| `attention.py` | Self-attention head |
| `multi_head_attention.py` | Multi-headed self-attention |
| `transformer.py` | Transformer block |
| `gpt.py` | GPT model |
| `kv_cache.py` | KV-Cache for fast inference |
| `grouped_query_attention.py` | Grouped query attention |

The GPT stack is pre-norm: token embeddings plus learned positions, then attention and a feed-forward block with a residual around each. The token embedding and the output head share one weight matrix. Pass `norm="rmsnorm"` to `GPTConfig` to swap layer norm for RMS norm. Grouped-query attention and the KV cache are the inference-side variants of the same attention math.
