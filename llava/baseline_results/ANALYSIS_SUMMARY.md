# Baseline Evaluation Analysis Summary

## Overall Performance

- **Total Test Samples**: 300
- **Overall Accuracy**: 68.67% (206/300 correct)
- **AUROC**: 0.6867
- **Precision**: 0.7642
- **Recall**: 0.5400
- **F1 Score**: 0.6328

## Performance by Class

### Safe Prompts (Label 0)
- **Total**: 150 samples
- **Accuracy**: 83.33% (125/150 correct)
- **Model correctly says "yes"**: 125
- **Model incorrectly says "no"**: 25

### Unsafe/Emergent Unsafe Prompts (Label 1)
- **Total**: 150 samples
- **Accuracy**: 54.00% (81/150 correct)
- **Model correctly says "no"**: 81
- **Model incorrectly says "yes"**: 69

## Does the Model Understand Emergent Unsafe?

**Answer: PARTIALLY - Better than random, but not great**

- The model correctly identifies emergent unsafe prompts **54% of the time**
- This is **better than random (50%)**, but far from perfect
- The model has a **46% false positive rate** (says "yes" when it should say "no" to unsafe prompts)

### Key Findings:
1. ✅ Model shows SOME understanding (54% > 50% random)
2. ❌ Model struggles with emergent unsafe detection (only 54% accuracy)
3. ✅ Model is much better at identifying safe prompts (83% accuracy)
4. ⚠️ Model is biased toward saying "yes" (more likely to answer than refuse)

## Confusion Matrix

```
Predicted:      YES      NO
Actual SAFE:    125      25
Actual UNSAFE:   69      81
```

## Common Error Patterns

### Model says "yes" to unsafe prompts (46% of unsafe cases):
- Prompts about dangerous experiments
- Prompts about testing limits/breaking things
- Prompts about recreating potentially unsafe scenarios

### Model says "no" to safe prompts (17% of safe cases):
- Simple factual questions
- Questions about image content
- Questions that seem ambiguous to the model

## Data Splits

### Train Split
- **File**: `train_split.jsonl`
- **Samples**: 1,395
- **Safe**: 704 (50.47%)
- **Unsafe**: 691 (49.53%)

### Test Split
- **File**: `test_split.jsonl`
- **Samples**: 300
- **Safe**: 150 (50.00%)
- **Unsafe**: 150 (50.00%)

### Validation Split
- **Samples**: 299
- **Safe**: 143 (47.83%)
- **Unsafe**: 156 (52.17%)

## Conclusion

The baseline LLaVA model has **limited understanding of emergent unsafe prompts**. While it performs better than random (54% vs 50%), it fails to identify nearly half of the unsafe prompts. This suggests that:

1. **Training is necessary**: The zero-shot baseline is insufficient
2. **LoRA training should help**: Fine-tuning on safety labels should improve detection
3. **The task is challenging**: Emergent unsafe requires understanding context + image together

The model is much better at identifying safe prompts (83% accuracy), suggesting it's generally conservative but struggles with the nuanced task of detecting when a safe-looking prompt becomes unsafe in context.
