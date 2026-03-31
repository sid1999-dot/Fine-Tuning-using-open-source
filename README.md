# 🧠 LLM Fine-Tuning & Quantization — Open Source Experiments

> A 2-week hands-on deep dive into fine-tuning open-source LLMs using LoRA, QLoRA, and quantization techniques.  
> Built while completing [Ed Donner's LLM / Agentic AI courses](https://www.udemy.com/user/ed-donner/) and pursuing a career transition into AI/GenAI Engineering.

---

## 📌 About This Repo

This repository documents my practical experiments with:
- **Quantization** (Full Precision → 8-bit → 4-bit NF4)
- **LoRA / QLoRA fine-tuning** on open-source models
- **PEFT (Parameter-Efficient Fine-Tuning)** using HuggingFace `peft` + `transformers`
- Model architecture inspection and adapter math

Everything here is real code I ran, with outputs and explanations — not just copy-pasted tutorials.

---

## 🗂️ Repository Structure

```
📦 llm-finetuning-experiments/
 ┣ 📓 Fine_tuning_open_source.ipynb     ← Week 1: Quantization + LoRA loading
 ┣ 📄 README.md                         ← This file (updated weekly)
 ┗ 📁 notebooks/                        ← Future notebooks added here
```

---

## 📅 Progress Log

### ✅ Week 1 — Quantization & LoRA Adapter Deep Dive
> **Model:** `meta-llama/Llama-3.2-3B` | **Notebook:** `Fine_tuning_open_source.ipynb`

**What I did:**
- Loaded Llama 3.2-3B in full precision, 8-bit, and 4-bit (NF4/QLoRA)
- Compared memory footprints across quantization levels
- Loaded a pre-trained LoRA adapter (`ed-donner/price-*`) on top of the 4-bit base
- Manually computed LoRA parameter counts to verify adapter size (~73.4 MB for r=32)
- Explored how rank (`r`) and target modules affect adapter size (r=256 + MLP → 1.5 GB)

**Key results:**

| Configuration | Memory Usage |
|---|---|
| Full precision (fp32) | 6.4 GB |
| 8-bit quantization | 3.6 GB |
| 4-bit NF4 (QLoRA) | 2.2 GB |
| 4-bit base + LoRA adapter | 2.27 GB |

**Key concepts learned:**
- `BitsAndBytesConfig` with `nf4` + `double_quant` for memory-efficient loading
- How `lora_A × lora_B` matrices inject trainable params without touching base weights
- Architecture diff: `Linear` → `Linear8bitLt` → `Linear4bit` → `lora.Linear4bit`
- Why LoRA rank (`r`) is the main lever controlling adapter size vs. quality tradeoff

---

### 🔜 Week 2 — *(Coming Soon)*
> Planned: Fine-tuning Qwen with LoRA from scratch, tracking experiments with W&B, pushing fine-tuned adapter to HuggingFace Hub

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `transformers` | Model loading, tokenization, training |
| `peft` | LoRA / QLoRA adapter management |
| `bitsandbytes` | 4-bit / 8-bit quantization |
| `torch` | Backend compute (CUDA) |
| Google Colab (T4 GPU) | Runtime environment |
| HuggingFace Hub | Model registry + adapter hosting |

---

## 🧩 Concepts Reference

### What is LoRA?
Low-Rank Adaptation. Instead of updating all model weights during fine-tuning, LoRA injects two small matrices (`lora_A`, `lora_B`) into target layers. The effective weight update is:

```
ΔW = α × (lora_A × lora_B)
```

Where `r` (rank) controls the bottleneck dimension. Lower rank = fewer params = faster training, but less expressive. Typical values: r=8 to r=64 for most tasks.

### What is QLoRA?
LoRA on top of a 4-bit quantized base model. The base model weights are frozen in 4-bit, but LoRA adapters are trained in bfloat16. This is how you fine-tune a 7B+ model on a single consumer GPU.

### Why NF4?
Normal Float 4-bit — designed specifically for weights that follow a normal (Gaussian) distribution. More accurate than plain INT4 for LLM weights.

---

## 🙋 About Me

**Siddharth Basu** — Associate Systems Engineer @ TCS, transitioning into AI/GenAI Engineering.

Working on: LLM evaluation pipelines, red teaming (XPIA), RAG systems, and agent frameworks.

- 🤗 HuggingFace: *(add your profile link)*
- 💼 LinkedIn: *(add your profile link)*
- 🐙 GitHub: *(this repo)*

---

## 📝 Notes

- Notebooks are run on Google Colab with T4 GPU (free tier)
- All model weights loaded from HuggingFace — requires `HF_TOKEN` in Colab secrets for gated models (Llama)
- This repo is updated continuously over a 2-week sprint — check back for new notebooks

---

*Last updated: Week 1 — March 2026*
