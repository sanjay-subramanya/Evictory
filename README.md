# Evictory

**Metacognitive KV cache eviction - keeping the unexpected, discarding the obvious**

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/sanjay-subramanya/Evictory)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Evictory is a lightweight inference‑time experiment that dynamically manages the KV cache of an LLM. Instead of keeping every token, it uses the model’s own prediction entropy (surprise) to decide what to evict. It also adapts its recency window based on loss volatility and protects entire conversation turns. This results in bounded memory growth without sacrificing coherence.

## How It Works

1. **Prediction entropy** – For each generated token, we compute the negative log likelihood (loss). Low loss = predictable token (e.g., “the”, “and”) → safe to evict. High loss = surprising token → protect.
2. **Loss volatility** – The standard deviation of the last N token losses. Low volatility (stable text) → shrink the recency window (evict more aggressively). High volatility (topic shifts, surprises) → expand the recency window (keep more recent context).
3. **Turn‑aware protection** – Tokens from previous conversation turns are never evicted. Only the current turn is compressed.
4. **Similarity‑guided eviction** – When the cache exceeds a soft limit, we find the most similar pair of tokens (cosine similarity of last‑layer key vectors). If their similarity exceeds a loss‑adaptive threshold, we evict the token with **lower** prediction entropy (the more predictable one). This removes redundancy without losing meaning.

No fine‑tuning, no external prompts, no extra model calls. All signals come from the model’s own forward pass.

## Key Features
- **Stateful chat** with streaming responses via Gradio
- **Live telemetry** - cache size, evictions, compression ratio, volatility, adaptive recency window
- **Adaptive recency** - window size automatically adjusts (4–16) based on loss volatility
- **Turn isolation** - previous turns are frozen, never evicted
- **Local‑first** - works offline after the model is downloaded

## Overview
<div align="center"> 

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KV Memory Manager (Eviction)                       │
├───────────────┬──────────────────────────────────────┬──────────────────────┤
│   Sink        │   Previous turns (fully protected)   │   Current turn       │
│  (4 tokens)   │   – never evicted                    │   – recency window   │
│               │                                      │     (4–16 tokens)    │
│               │                                      │   – similarity‑based │
│               │                                      │     eviction         │
└───────────────┴──────────────────────────────────────┴──────────────────────┘
```
</div>


## Configuration
All options live in `config/settings.py`:
- `base_model` - Default: `Qwen/Qwen2.5-0.5B-Instruct`
- `model_path` - Where the model is stored locally (Default: `models/base` inside the repo)
- `device` - `"cpu"` by default; set to `"cuda"` if you have a GPU
- `dtype` - Inferred from device (BF16 on CPU, FP16 on CUDA)
- `base_threshold` - Cosine similarity threshold for eviction (default 0.9)
- `loss_scale` - How much prediction loss influences the threshold (default 0.35)
- `volatility_window` - Number of recent losses to compute volatility (default 10)
- `volatility_update_interval` - Steps between volatility recalculations (default 10)
- `max_new_tokens` - Maximum tokens to generate per response (default 150)

> Note: By default the decoder uses `local_files_only=True` (fast/offline after download). To skip pre-download and fetch on demand, set `local_files_only=False` for both the tokenizer and model in `core/decoder.py`. This requires internet access (and an HF token if the model you want to use is private) and may increase first-run latency.

## Trade-offs & Design Philosophy

**Memory vs. Coherence**  
Evicting tokens based on prediction entropy can remove low‑surprise but context‑critical tokens (e.g., a rare name that becomes predictable after a few mentions). However, the combination of turn‑aware protection and similarity‑guided eviction (only drop near‑duplicates) keeps coherence high. Empirical tests show this approach outperforms pure sliding window on long‑form generation.

**No Fine-Tuning Required**  
All signals (prediction loss, hidden states, attention) are already computed during standard generation. The system works with any causal LM that returns logits and (optionally) attention weights. It requires `attn_implementation="eager"` to access attention.

**What You Gain vs. What You Sacrifice**  
- **Gain:** Bounded memory growth, adaptive recency, preservation of surprising tokens, long‑turn coherence
- **Sacrifice:** Small overhead for loss/volatility computation and occasional eviction of predictable but useful tokens
- **No sacrifice:** Model weights, task performance, or need for retraining


## Quickstart (Local)
0) Configure (optional)  
   Adjust `config/settings.py` to change base model, model path, device, or cache windows.

1) Install the necessary packages:

    ```bash
    pip install -r requirements.txt
    ``` 

2) Download the base model  
This uses `huggingface_hub.snapshot_download` and saves to `models/base` by default.
   
    ```bash
    python download.py
    ```
    Alternatively (no pre-download): set `local_files_only=False` in `core/decoder.py` for both `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`. The model and tokenizer will be fetched automatically on first run.

3) Launch the Gradio app  
    ```bash
    python app.py
    ```
    The server will be accessible at localhost:7860 by default.


## Benchmarks
This compares the performance of Evictory's approach with the standard model downloaded earlier.
A small set of prompts is provided in `benchmark/test_prompts.py` to act as a baseline. You can add more prompts to this file to test against different scenarios.

  ```bash
  python run_benchmark.py
  ```
Results are written to `benchmark/results/` by default.

## Tips & Troubleshooting
- Model not found / load fails:  
  Make sure you ran `python download.py` (locally) or configured a pre-build step (Spaces).
- Long, rambling answers:  
  Lower `max_new_tokens` or tweak the decoding temperature/top-k in `decoder.py`.
  