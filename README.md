# CS2420 Project: Emergent Unsafe Prompts in Vision-Language Models

This repository contains the implementation and evaluation of methods for generating and detecting emergent unsafe prompts in vision-language models (VLMs).

## 📁 Project Structure

```
cs2420-project/
├── pipeline1/                          # Initial two-stage prompt generation
├── pipeline2/                          # Caption-based prompt generation scripts
├── pipeline2_caption_to_prompt/       # Organized caption-to-prompt pipeline with full data
├── pipeline3/                          # Advanced prompt generation and evaluation
├── llava/                             # LLaVA model training and evaluation
├── lora/                              # LoRA fine-tuning infrastructure
├── final_evals/                       # Final evaluation scripts and results
└── coco_val_sample-SPECIFIC/          # COCO validation image subset (500 images)
```

## 🔬 Pipeline Overview

### Pipeline 1: Two-Stage Emergence
**Location:** `pipeline1/`

Initial approach using a two-stage method for generating emergent unsafe prompts.

**Key Scripts:**
- `two_stage_emerge.py` - Two-stage prompt generation
- `evaluate_results.py` - Evaluation framework

### Pipeline 2: Caption-to-Prompt Generation
**Location:** `pipeline2/` (scripts) + `pipeline2_caption_to_prompt/` (full implementation with data)

Systematic prompt generation using multi-level image captions with frontier models (Claude Haiku 3.5, Gemini 2.5 Flash Lite).

**Key Scripts:**
- `coco_caption_pipeline.py` - Multi-level caption generation
- `live_caption_eval.py` - Live evaluation pipeline
- `evaluate_results.py` - LLM judge evaluation using Gemini
- `analyze_voting_vs_llm.py` - Compare human annotations vs LLM judge
- `emerge.py` - Core emergence utilities
- `prompts.py` - Prompt templates

**Organized Data Structure:** `pipeline2_caption_to_prompt/processed_results/`
- `scripts/` - All Python processing scripts
- `data/` - Prompts, evaluations, and samples
- `human_annotations/` - Voting results (101 prompts, 28 images, 4 annotators)
- `analysis/` - Metrics and summaries
- `archive/` - Legacy files
- See [detailed README](pipeline2_caption_to_prompt/processed_results/README.md)

### Pipeline 3: Advanced Evaluation
**Location:** `pipeline3/`

Extended prompt generation with additional image categories and evaluation methods.

**Key Scripts:**
- `eval.py` / `eval-update.py` - Comprehensive evaluation
- `evaluate_prompts.py` - Prompt assessment
- `download_coco_images.py` - COCO dataset utilities
- `download_human_animal_images.py` - Specific category downloads
- `copy_random_images.py` - Random sampling utilities

### LoRA: Low-Rank Adaptation Training
**Location:** `lora/`

LoRA fine-tuning infrastructure for LLaMA models to generate context-aware emergent unsafe prompts.

**Structure:**
- `scripts/` - Training and evaluation scripts
- `data/` - Training datasets
- See [LoRA README](lora/README.md) for details


### LLaVA: Vision-Language Model Training
**Location:** `llava/`

LLaVA-based model training and evaluation for emergent unsafe prompt detection.

**Key Scripts:**
- `train_lora.py` - LoRA training for LLaVA
- `evaluate_lora.py` - Model evaluation
- `create_balanced_splits.py` - Dataset splitting
- `verify_splits.py` - Data validation
- `generate_comparison_plots.py` - Visualization

**SLURM Scripts:**
- `run_lora_training.sbatch` - Distributed training
- `run_lora_eval.sbatch` - Batch evaluation

### Final Evaluations
**Location:** `final_evals/`

Production-ready evaluation scripts using Claude and Gemini.

**Scripts:**
- `eval-claude.py` - Claude-based evaluation
- `eval-gemini.py` - Gemini-based evaluation

**Results:**
- `claude_eval.jsonl` - Claude evaluation results (~4MB)
- `gemini_eval.jsonl` - Gemini evaluation results (~3.6MB)

## 📊 Key Results

### Human Evaluation (101 prompts across 28 images)
- **38.6% of prompts** labeled unsafe by human majority vote (≥3/4 annotators)
- **89.3% of images** had at least one unsafe prompt
- **72% inter-annotator agreement** (moderate-substantial)

### LLM Judge Performance (Gemini 2.5 Flash Lite)
- **Accuracy:** 62.4%
- **Precision:** 50.8% (high over-flagging)
- **Recall:** 84.6% (catches most unsafe prompts)
- **False Positive Rate:** 51.6%
- **False Negative Rate:** 15.4%

## 🚀 Getting Started

### Prerequisites
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file with:
```bash
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_claude_api_key
```

### Running Pipeline 2 (Recommended)
```bash
cd pipeline2_caption_to_prompt/processed_results

# Generate prompts
python3 scripts/coco_caption_pipeline.py \
  --captions path/to/captions.json \
  --images-dir path/to/images \
  --output data/prompts/output.jsonl

# Evaluate with LLM judge
python3 scripts/evaluate_results.py \
  --results-file data/prompts/output.jsonl

# Compare with human annotations
python3 scripts/analyze_voting_vs_llm.py
```

## 📝 Documentation

- **Pipeline 2 Detailed Docs:** [pipeline2_caption_to_prompt/processed_results/README.md](pipeline2_caption_to_prompt/processed_results/README.md)
- **LoRA Training Guide:** [lora/README.md](lora/README.md)
- **Human Evaluation Summary:** [pipeline2_caption_to_prompt/processed_results/analysis/100_random_sample_results_summary.md](pipeline2_caption_to_prompt/processed_results/analysis/100_random_sample_results_summary.md)

## 🗂️ Data Organization

### COCO Images
- **Location:** `coco_val_sample-SPECIFIC/`
- **Size:** 500 images from COCO 2017 validation set
- **Coverage:** Diverse scenes including humans, animals, vehicles, indoor/outdoor

### Generated Prompts
- **Pipeline 2 prompts:** `pipeline2_caption_to_prompt/processed_results/data/prompts/`
- **Evaluation results:** `pipeline2_caption_to_prompt/processed_results/data/evaluations/`
- **Human annotations:** `pipeline2_caption_to_prompt/processed_results/human_annotations/`

### Analysis Results
- **Combined metrics:** `pipeline2_caption_to_prompt/processed_results/analysis/combined_metrics.json`
- **Summary reports:** `pipeline2_caption_to_prompt/processed_results/analysis/*.md`

## 🤝 Team

**Team 15:**
- Gardenia Liu
- Ryan Liu
- Keyu Wang
- Sarah Liaw

## 📄 Related Files

- **Project Report:** `CS2420_Fall25_Report_Team_15.tex`
- **Presentation Materials:** (if available)

## 🔐 Safety Note

This project is for academic research purposes on defensive AI security. The generated emergent unsafe prompts are used to:
1. Understand vulnerabilities in vision-language models
2. Develop better safety classifiers
3. Improve content moderation systems

All work follows responsible disclosure practices.

## 🔗 References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [LLaVA: Large Language and Vision Assistant](https://llava-vl.github.io/)
- [COCO Dataset](https://cocodataset.org/)

---

**Last Updated:** December 10, 2025
**Course:** CS2420 - Fall 2025
**Institution:** Harvard University
