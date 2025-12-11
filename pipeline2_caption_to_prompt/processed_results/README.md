# Processed Results Directory

This directory contains all data, scripts, and analysis results for the emergent unsafe prompts project.

## Directory Structure

```
processed_results/
├── README.md                          # This file
├── scripts/                           # Python scripts for processing
├── data/                              # Raw and processed datasets
├── human_annotations/                 # Human voting results
├── analysis/                          # Analysis results and metrics
├── images/                            # Image references
└── archive/                           # Old/deprecated files
```

## 📁 Subdirectory Details

### `scripts/`
Python scripts for data processing, evaluation, and analysis.

| Script | Purpose |
|--------|---------|
| `analyze_prompts.py` | Analyze generated prompts and compute statistics |
| `analyze_voting_vs_llm.py` | Compare human voting results against LLM judge predictions |
| `coco_caption_pipeline.py` | Generate multi-level captions from COCO images |
| `emerge.py` | Utility functions for emergent unsafe prompt generation |
| `evaluate_results.py` | Evaluate prompts using Gemini API as judge |
| `live_caption_eval.py` | Live evaluation pipeline for caption-prompt pairs |
| `prompts.py` | Prompt templates and generation utilities |

**Usage Example:**
```bash
# Run from project root
cd /Users/ryanliu/cs2420-project

# Analyze voting vs LLM judge
python3 pipeline2_caption_to_prompt/processed_results/scripts/analyze_voting_vs_llm.py

# Evaluate results with custom file
python3 pipeline2_caption_to_prompt/processed_results/scripts/evaluate_results.py \
  --results-file data/prompts/custom_prompts.jsonl
```

### `data/`
Contains all datasets organized by type.

#### `data/prompts/`
Generated prompts from various pipelines.

| File | Description | Size |
|------|-------------|------|
| `coco_caption_prompts-v1-merged.jsonl` | Merged v1 prompts from all chunks | ~335KB |
| `1024_dataset_prompts.jsonl` | 1024-sample dataset prompts | ~428KB |
| `1300_gemini_v2_input_prompts.jsonl` | Gemini v2 generated prompts | ~994KB |
| `remaining_prompts.jsonl` | Continuation prompts | ~316KB |

#### `data/evaluations/`
LLM judge evaluations of generated prompts.

| File | Description | Size |
|------|-------------|------|
| `live_caption_eval-v2.jsonl` | Main v2 evaluation results | ~1.2MB |
| `live_caption_eval-v2-formatted.jsonl` | Formatted v2 evaluations | ~1.2MB |
| `1024_dataset_evals.json` | Evaluations for 1024 dataset | ~1.0MB |
| `remaining_eval.jsonl` | Continuation evaluations | ~308KB |

#### `data/samples/`
Random samples used for human evaluation.

| File | Description |
|------|-------------|
| `50_random_sample.jsonl` | First 50-prompt random sample |
| `50_random_sample_the_second.jsonl` | Second 49-prompt random sample |

### `human_annotations/`
Human voting results and voting interface.

| File | Description | Annotators | Date |
|------|-------------|------------|------|
| `voting_interface.html` | Web interface for human annotation | - | - |
| `voting_results_2025-11-30-0.json` | Vote batch 1 | Annotator 0 | Nov 30 |
| `voting_results_2025-11-30-1.json` | Vote batch 2 | Annotator 1 | Nov 30 |
| `voting_results_2025-11-30-2.json` | Vote batch 3 | Annotator 2 | Nov 30 |
| `voting_results_2025-11-30-3.json` | Vote batch 4 | Annotator 3 | Nov 30 |
| `voting_results_2025-12-04.json` | Second round votes | Annotator 0 | Dec 04 |
| `voting_results_2025-12-05.json` | Second round votes | Annotator 1 | Dec 05 |
| `voting_results_2025-12-08.json` | Second round votes | Annotator 2 | Dec 08 |
| `voting_results_2025-12-08-2.json` | Second round votes | Annotator 3 | Dec 08 |

**Total: 101 prompts across 28 images with 4-way voting**

### `analysis/`
Analysis results, metrics, and summaries.

| File | Description |
|------|-------------|
| `combined_metrics.json` | Combined metrics from all evaluations (101 prompts) |
| `voting_vs_llm_analysis.json` | Comparison between human votes and LLM judge |
| `voting_vs_llm_analysis_COMPLETE.json` | Complete analysis with all details |
| `voting_vs_llm_IMAGE_LEVEL.json` | Image-level aggregated analysis |
| `100_random_sample_results_summary.md` | **Main summary report** with prompt & image stats |
| `50_random_sample_results_summary.md` | Summary of first 50-sample batch |

**Key Metrics (from 100_random_sample_results_summary.md):**
- **Prompt-level:** 101 prompts, 39 unsafe (38.6%), 62 safe (61.4%)
- **Image-level:** 28 images, 25 (89.3%) with ≥1 unsafe prompt
- **LLM Judge:** 62.4% accuracy, 50.8% precision, 84.6% recall
- **False Positive Rate:** 51.6% (LLM over-flags safe prompts)
- **False Negative Rate:** 15.4% (LLM misses some unsafe prompts)

### `images/`
References to COCO validation images.

| Item | Description |
|------|-------------|
| `coco_val_sample-SPECIFIC/` | Directory with matched COCO validation images |
| `coco_val_sample-SPECIFIC-500` | Symlink to 500-image subset |

### `archive/`
Old, intermediate, and deprecated files (kept for reference).

#### `archive/chunks/`
Old chunked prompt files from initial pipeline runs.
- `coco_caption_prompts-v1-chunk0.jsonl` through `chunk9.jsonl`
- `coco_caption_prompts.jsonl` (legacy)
- Various empty/incomplete runs

#### `archive/intermediate/`
Intermediate evaluation and processing files.
- `50_random_sample_evals.jsonl` - Initial evaluations
- `50_random_sample_evals_complete.json` - Completed first batch
- `50_random_sample_the_second_*.jsonl` - Second batch intermediate files
- `live_caption_eval-v2-*.jsonl` - Versioned evaluation snapshots

#### `archive/legacy/`
Legacy evaluation versions (v1, v3, v4, v5, v6).
- `live_caption_eval-v1.jsonl` through `v6.jsonl`
- `out.json`, `test_captions_subset.json` - Test files

## 🔄 Data Flow

```
1. Image Collection
   ↓
   images/coco_val_sample-SPECIFIC/

2. Caption Generation
   ↓
   scripts/coco_caption_pipeline.py
   ↓
   data/prompts/coco_caption_prompts-v1-merged.jsonl

3. Prompt Generation (Pipeline 1: Frontier Models)
   ↓
   data/prompts/1300_gemini_v2_input_prompts.jsonl

4. LLM Judge Evaluation
   ↓
   scripts/evaluate_results.py
   ↓
   data/evaluations/live_caption_eval-v2.jsonl

5. Random Sampling
   ↓
   data/samples/50_random_sample.jsonl

6. Human Annotation
   ↓
   human_annotations/voting_interface.html
   ↓
   human_annotations/voting_results_2025-*.json

7. Analysis & Comparison
   ↓
   scripts/analyze_voting_vs_llm.py
   ↓
   analysis/combined_metrics.json
   analysis/100_random_sample_results_summary.md
```

## 📊 Key Findings

### Human Evaluation (101 prompts, 28 images)
- **38.6% of prompts** were labeled unsafe by human majority vote (≥3/4 annotators)
- **89.3% of images** had at least one unsafe prompt
- **72% inter-annotator agreement** (moderate-substantial)

### LLM Judge Performance
- **Accuracy:** 62.4% when compared to human labels
- **High recall (84.6%)** but low precision (50.8%)
- **Major issue:** Over-flagging safe prompts (51.6% false positive rate)
- **Strength:** Rarely misses truly unsafe prompts (15.4% false negative rate)

### Common False Positive Patterns
1. Playful physics questions ("How high can we throw it?")
2. Photography/observational prompts ("Capture the precise moment")
3. Household scenarios (blender, fireplace, toilet)
4. Hypothetical speculation ("I wonder if...")

## 🛠️ Maintenance Notes

### Adding New Data
1. **New prompts:** Add to `data/prompts/`
2. **New evaluations:** Add to `data/evaluations/`
3. **New voting results:** Add to `human_annotations/`
4. **New analysis:** Add to `analysis/`

### Running Scripts After Reorganization
All scripts now use relative paths from the `processed_results` directory root:
- `scripts/` for Python files
- `data/` for input/output data
- `analysis/` for results

If running from project root, use full paths:
```bash
python3 pipeline2_caption_to_prompt/processed_results/scripts/analyze_voting_vs_llm.py
```

### Deprecated Files
Old files are in `archive/` but not deleted. They can be safely removed if:
1. Confirmed no longer needed for reproduction
2. Newer versions exist in main directories
3. Documented in this README

## 📝 Version History

- **Dec 10, 2024:** Major reorganization - separated scripts, data, annotations, and analysis
- **Dec 08, 2024:** Second batch of human annotations completed (49 prompts)
- **Dec 05, 2024:** Combined metrics from both annotation batches
- **Nov 30, 2024:** First batch of human annotations (52 prompts)
- **Nov 29, 2024:** Initial data collection and prompt generation

## 🔗 Related Files

- **Project root:** `/Users/ryanliu/cs2420-project/`
- **Report:** `CS2420_Fall25_Report_Team_15.tex`
- **COCO images:** `coco_val_sample-SPECIFIC-500/`

## 👥 Contributors

- Gardenia Harris
- Ryan Liu
- Keyu Chen
- Sarah Johnson

---

**Last Updated:** December 10, 2024
**Maintained by:** Team 15
