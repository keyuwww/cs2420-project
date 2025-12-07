#!/usr/bin/env python3
"""
Compare Baseline vs LoRA Results

Loads metrics from both evaluations and creates a comparison report.
"""

import json
from pathlib import Path
import argparse

def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_metrics(baseline_metrics, lora_metrics):
    """Compare metrics and calculate improvements"""
    comparison = {}
    
    metrics_to_compare = [
        'accuracy', 'auroc', 'precision', 'recall', 'f1',
        'safe_accuracy', 'unsafe_accuracy'
    ]
    
    for metric in metrics_to_compare:
        if metric in baseline_metrics and metric in lora_metrics:
            baseline_val = baseline_metrics[metric]
            lora_val = lora_metrics[metric]
            improvement = lora_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            
            comparison[metric] = {
                'baseline': baseline_val,
                'lora': lora_val,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            }
    
    return comparison

def print_comparison(comparison):
    """Print formatted comparison"""
    print("\n" + "="*70)
    print("BASELINE vs LORA COMPARISON")
    print("="*70)
    print(f"{'Metric':<20} {'Baseline':<12} {'LoRA':<12} {'Improvement':<15} {'% Change':<10}")
    print("-"*70)
    
    for metric, values in comparison.items():
        baseline = values['baseline']
        lora = values['lora']
        improvement = values['improvement']
        pct = values['improvement_pct']
        
        # Format improvement with sign
        improvement_str = f"{improvement:+.4f}"
        pct_str = f"{pct:+.2f}%"
        
        print(f"{metric:<20} {baseline:<12.4f} {lora:<12.4f} {improvement_str:<15} {pct_str:<10}")
    
    print("="*70)

def compare_confusion_matrices(baseline_cm, lora_cm):
    """Compare confusion matrices"""
    print("\n" + "="*70)
    print("CONFUSION MATRIX COMPARISON")
    print("="*70)
    print("\nBaseline Confusion Matrix:")
    print(f"  Predicted:      YES      NO")
    print(f"  Actual SAFE:   {baseline_cm[0][0]:4d}    {baseline_cm[0][1]:4d}")
    print(f"  Actual UNSAFE: {baseline_cm[1][0]:4d}    {baseline_cm[1][1]:4d}")
    
    print("\nLoRA Confusion Matrix:")
    print(f"  Predicted:      YES      NO")
    print(f"  Actual SAFE:   {lora_cm[0][0]:4d}    {lora_cm[0][1]:4d}")
    print(f"  Actual UNSAFE: {lora_cm[1][0]:4d}    {lora_cm[1][1]:4d}")
    
    print("\nImprovements:")
    print(f"  True Negatives (Safe→Safe):   {lora_cm[0][0] - baseline_cm[0][0]:+4d}")
    print(f"  False Positives (Safe→Unsafe): {lora_cm[0][1] - baseline_cm[0][1]:+4d}")
    print(f"  False Negatives (Unsafe→Safe): {lora_cm[1][0] - baseline_cm[1][0]:+4d}")
    print(f"  True Positives (Unsafe→Unsafe): {lora_cm[1][1] - baseline_cm[1][1]:+4d}")

def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs LoRA results")
    parser.add_argument("--baseline-metrics", type=str,
                       default="baseline_results/baseline_metrics_test.json",
                       help="Path to baseline metrics JSON")
    parser.add_argument("--lora-metrics", type=str,
                       default="lora_results/lora_metrics_test.json",
                       help="Path to LoRA metrics JSON")
    parser.add_argument("--output", type=str, default="comparison_results/comparison.json",
                       help="Output path for comparison JSON")
    
    args = parser.parse_args()
    
    # Load metrics
    baseline_path = Path(args.baseline_metrics)
    lora_path = Path(args.lora_metrics)
    
    if not baseline_path.exists():
        print(f"Error: Baseline metrics not found at {baseline_path}")
        return
    
    if not lora_path.exists():
        print(f"Error: LoRA metrics not found at {lora_path}")
        return
    
    print(f"Loading baseline metrics from {baseline_path}")
    baseline_metrics = load_metrics(baseline_path)
    
    print(f"Loading LoRA metrics from {lora_path}")
    lora_metrics = load_metrics(lora_path)
    
    # Compare metrics
    comparison = compare_metrics(baseline_metrics, lora_metrics)
    
    # Print comparison
    print_comparison(comparison)
    
    # Compare confusion matrices
    if 'confusion_matrix' in baseline_metrics and 'confusion_matrix' in lora_metrics:
        compare_confusion_matrices(
            baseline_metrics['confusion_matrix'],
            lora_metrics['confusion_matrix']
        )
    
    # Save comparison
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    comparison_data = {
        'baseline_metrics': baseline_metrics,
        'lora_metrics': lora_metrics,
        'comparison': comparison,
        'summary': {
            'accuracy_improvement': comparison['accuracy']['improvement'],
            'auroc_improvement': comparison['auroc']['improvement'],
            'f1_improvement': comparison['f1']['improvement'],
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✓ Comparison saved to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Accuracy improvement: {comparison['accuracy']['improvement']:+.4f} ({comparison['accuracy']['improvement_pct']:+.2f}%)")
    print(f"AUROC improvement: {comparison['auroc']['improvement']:+.4f} ({comparison['auroc']['improvement_pct']:+.2f}%)")
    print(f"F1 improvement: {comparison['f1']['improvement']:+.4f} ({comparison['f1']['improvement_pct']:+.2f}%)")
    print("="*70)

if __name__ == "__main__":
    main()


