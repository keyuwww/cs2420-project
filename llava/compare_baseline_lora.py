#!/usr/bin/env python3
"""
Compare Baseline vs LoRA Evaluation Results

Generates detailed comparison of baseline and LoRA evaluation results,
including side-by-side metrics, confusion matrices, and error analysis.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from sklearn.metrics import confusion_matrix, classification_report

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics from JSON file"""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_results(results_path: str) -> List[Dict]:
    """Load detailed results from JSONL file"""
    results = []
    with open(results_path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compare_metrics(baseline_metrics: Dict, lora_metrics: Dict, output_dir: Path):
    """Compare metrics and create comparison plots"""
    
    metrics_to_compare = [
        ('accuracy', 'Accuracy'),
        ('auroc', 'AUROC'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1', 'F1 Score'),
        ('safe_accuracy', 'Safe Accuracy'),
        ('unsafe_accuracy', 'Unsafe Accuracy'),
    ]
    
    # Create comparison table
    print("\n" + "="*80)
    print("METRICS COMPARISON: Baseline vs LoRA")
    print("="*80)
    print(f"{'Metric':<20} {'Baseline':<15} {'LoRA':<15} {'Difference':<15} {'Status':<10}")
    print("-"*80)
    
    differences = []
    for metric_key, metric_name in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric_key, 0)
        lora_val = lora_metrics.get(metric_key, 0)
        diff = lora_val - baseline_val
        diff_pct = (diff / baseline_val * 100) if baseline_val > 0 else 0
        
        if abs(diff) < 1e-6:
            status = "IDENTICAL"
        elif diff > 0:
            status = "IMPROVED"
        else:
            status = "DECREASED"
        
        differences.append({
            'metric': metric_name,
            'baseline': baseline_val,
            'lora': lora_val,
            'diff': diff,
            'diff_pct': diff_pct,
            'status': status
        })
        
        print(f"{metric_name:<20} {baseline_val:<15.4f} {lora_val:<15.4f} {diff:+.4f} ({diff_pct:+.2f}%) {status:<10}")
    
    print("="*80)
    
    # Create bar plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall metrics comparison
    ax = axes[0, 0]
    metrics_names = [m['metric'] for m in differences[:5]]
    baseline_vals = [m['baseline'] for m in differences[:5]]
    lora_vals = [m['lora'] for m in differences[:5]]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, lora_vals, width, label='LoRA', alpha=0.8, color='#e74c3c')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Class-specific accuracy comparison
    ax = axes[0, 1]
    class_metrics = differences[5:]
    class_names = [m['metric'] for m in class_metrics]
    baseline_class = [m['baseline'] for m in class_metrics]
    lora_class = [m['lora'] for m in class_metrics]
    
    x = np.arange(len(class_names))
    bars1 = ax.bar(x - width/2, baseline_class, width, label='Baseline', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, lora_class, width, label='LoRA', alpha=0.8, color='#e74c3c')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Class-Specific Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    # Confusion matrix comparison
    baseline_cm = np.array(baseline_metrics['confusion_matrix'])
    lora_cm = np.array(lora_metrics['confusion_matrix'])
    
    # Baseline confusion matrix
    ax = axes[1, 0]
    sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['YES (Safe)', 'NO (Unsafe)'],
                yticklabels=['Safe', 'Unsafe'],
                cbar_kws={'label': 'Count'})
    ax.set_title('Baseline Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    
    # LoRA confusion matrix
    ax = axes[1, 1]
    sns.heatmap(lora_cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=['YES (Safe)', 'NO (Unsafe)'],
                yticklabels=['Safe', 'Unsafe'],
                cbar_kws={'label': 'Count'})
    ax.set_title('LoRA Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    
    plt.tight_layout()
    comparison_plot_path = output_dir / 'comparison_metrics.png'
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {comparison_plot_path}")
    plt.close()
    
    return differences


def compare_predictions(baseline_results: List[Dict], lora_results: List[Dict], output_dir: Path):
    """Compare individual predictions and identify differences"""
    
    # Create dictionaries for easy lookup
    baseline_dict = {r['image_path']: r for r in baseline_results}
    lora_dict = {r['image_path']: r for r in lora_results}
    
    # Find differences
    same_predictions = 0
    different_predictions = 0
    baseline_correct_lora_wrong = []
    baseline_wrong_lora_correct = []
    both_wrong_differently = []
    
    for image_path in baseline_dict.keys():
        if image_path not in lora_dict:
            continue
        
        baseline = baseline_dict[image_path]
        lora = lora_dict[image_path]
        
        baseline_pred = baseline['prediction']
        lora_pred = lora['prediction']
        true_label = baseline['ground_truth']
        
        baseline_correct = (baseline_pred == true_label)
        lora_correct = (lora_pred == true_label)
        
        if baseline_pred == lora_pred:
            same_predictions += 1
        else:
            different_predictions += 1
            
            if baseline_correct and not lora_correct:
                baseline_correct_lora_wrong.append({
                    'image_path': image_path,
                    'prompt': baseline['prompt'],
                    'true_label': true_label,
                    'baseline_pred': baseline_pred,
                    'lora_pred': lora_pred,
                    'baseline_response': baseline.get('response', '')[:100],
                    'lora_response': lora.get('response', '')[:100]
                })
            elif not baseline_correct and lora_correct:
                baseline_wrong_lora_correct.append({
                    'image_path': image_path,
                    'prompt': baseline['prompt'],
                    'true_label': true_label,
                    'baseline_pred': baseline_pred,
                    'lora_pred': lora_pred,
                    'baseline_response': baseline.get('response', '')[:100],
                    'lora_response': lora.get('response', '')[:100]
                })
            else:
                both_wrong_differently.append({
                    'image_path': image_path,
                    'prompt': baseline['prompt'],
                    'true_label': true_label,
                    'baseline_pred': baseline_pred,
                    'lora_pred': lora_pred
                })
    
    print("\n" + "="*80)
    print("PREDICTION AGREEMENT ANALYSIS")
    print("="*80)
    print(f"Total samples: {len(baseline_dict)}")
    print(f"Same predictions: {same_predictions} ({same_predictions/len(baseline_dict)*100:.2f}%)")
    print(f"Different predictions: {different_predictions} ({different_predictions/len(baseline_dict)*100:.2f}%)")
    print(f"\nBaseline correct, LoRA wrong: {len(baseline_correct_lora_wrong)}")
    print(f"Baseline wrong, LoRA correct: {len(baseline_wrong_lora_correct)}")
    print(f"Both wrong, but differently: {len(both_wrong_differently)}")
    
    # Save detailed comparison
    comparison_data = {
        'summary': {
            'total_samples': len(baseline_dict),
            'same_predictions': same_predictions,
            'different_predictions': different_predictions,
            'baseline_correct_lora_wrong': len(baseline_correct_lora_wrong),
            'baseline_wrong_lora_correct': len(baseline_wrong_lora_correct),
            'both_wrong_differently': len(both_wrong_differently)
        },
        'baseline_correct_lora_wrong': baseline_correct_lora_wrong[:20],  # First 20 examples
        'baseline_wrong_lora_correct': baseline_wrong_lora_correct[:20],
        'both_wrong_differently': both_wrong_differently[:20]
    }
    
    comparison_path = output_dir / 'prediction_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\n✓ Saved detailed comparison to {comparison_path}")
    
    return comparison_data


def generate_summary_report(baseline_metrics: Dict, lora_metrics: Dict, 
                           differences: List[Dict], comparison_data: Dict,
                           output_dir: Path):
    """Generate a comprehensive summary report"""
    
    report_path = output_dir / 'COMPARISON_REPORT.md'
    
    with open(report_path, 'w') as f:
        f.write("# Baseline vs LoRA Evaluation Comparison Report\n\n")
        f.write("## Executive Summary\n\n")
        
        # Check if metrics are identical
        all_identical = all(abs(d['diff']) < 1e-6 for d in differences)
        
        if all_identical:
            f.write("⚠️ **CRITICAL FINDING: LoRA and Baseline metrics are IDENTICAL**\n\n")
            f.write("This suggests that:\n")
            f.write("1. The LoRA adapter may not be properly loaded or applied\n")
            f.write("2. The LoRA training may not have converged or learned meaningful patterns\n")
            f.write("3. The evaluation may be using the base model instead of the LoRA-enhanced model\n")
            f.write("4. The LoRA adapter weights may be too small to affect the output\n\n")
        else:
            f.write("✅ LoRA shows differences from baseline, indicating the adapter is being used.\n\n")
        
        f.write("## Metrics Comparison\n\n")
        f.write("| Metric | Baseline | LoRA | Difference | Status |\n")
        f.write("|--------|----------|------|------------|--------|\n")
        
        for d in differences:
            status_emoji = "✅" if d['status'] == "IMPROVED" else "❌" if d['status'] == "DECREASED" else "⚪"
            f.write(f"| {d['metric']} | {d['baseline']:.4f} | {d['lora']:.4f} | "
                   f"{d['diff']:+.4f} ({d['diff_pct']:+.2f}%) | {status_emoji} {d['status']} |\n")
        
        f.write("\n## Prediction Agreement\n\n")
        f.write(f"- **Total Samples**: {comparison_data['summary']['total_samples']}\n")
        f.write(f"- **Same Predictions**: {comparison_data['summary']['same_predictions']} "
               f"({comparison_data['summary']['same_predictions']/comparison_data['summary']['total_samples']*100:.2f}%)\n")
        f.write(f"- **Different Predictions**: {comparison_data['summary']['different_predictions']} "
               f"({comparison_data['summary']['different_predictions']/comparison_data['summary']['total_samples']*100:.2f}%)\n")
        
        f.write("\n## Key Findings\n\n")
        
        if all_identical:
            f.write("### ⚠️ No Improvement Detected\n\n")
            f.write("The LoRA adapter appears to have no effect on the model's predictions. "
                   "This could indicate:\n\n")
            f.write("1. **LoRA not properly applied**: Check that `PeftModel.from_pretrained()` is working correctly\n")
            f.write("2. **Training issue**: The LoRA weights may not have learned meaningful patterns\n")
            f.write("3. **Evaluation issue**: The evaluation script may be using the base model\n")
            f.write("4. **LoRA configuration**: The LoRA rank/alpha may be too small to affect outputs\n\n")
        else:
            improvements = [d for d in differences if d['status'] == 'IMPROVED']
            if improvements:
                f.write("### ✅ Improvements\n\n")
                for imp in improvements:
                    f.write(f"- **{imp['metric']}**: Improved by {imp['diff']:.4f} "
                           f"({imp['diff_pct']:+.2f}%)\n")
                f.write("\n")
            
            decreases = [d for d in differences if d['status'] == 'DECREASED']
            if decreases:
                f.write("### ❌ Regressions\n\n")
                for dec in decreases:
                    f.write(f"- **{dec['metric']}**: Decreased by {abs(dec['diff']):.4f} "
                           f"({dec['diff_pct']:+.2f}%)\n")
                f.write("\n")
        
        f.write("## Recommendations\n\n")
        
        if all_identical:
            f.write("1. **Verify LoRA Loading**: Check the evaluation script to ensure LoRA adapter is loaded\n")
            f.write("2. **Check Training**: Review training logs to ensure LoRA weights were updated\n")
            f.write("3. **Inspect Checkpoints**: Verify that checkpoint files contain LoRA weights\n")
            f.write("4. **Increase LoRA Rank**: Try increasing `lora_rank` to make adapter more influential\n")
            f.write("5. **Debug Model State**: Print model parameters to verify LoRA is applied\n\n")
        else:
            f.write("1. **Analyze Improvements**: Review cases where LoRA improved predictions\n")
            f.write("2. **Address Regressions**: Investigate cases where LoRA performed worse\n")
            f.write("3. **Hyperparameter Tuning**: Adjust LoRA rank, alpha, or target modules\n")
            f.write("4. **Training Duration**: Consider training for more epochs if improvements are small\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `comparison_metrics.png`: Side-by-side metrics and confusion matrices\n")
        f.write("- `prediction_comparison.json`: Detailed prediction-level comparison\n")
        f.write("- `COMPARISON_REPORT.md`: This report\n\n")
    
    print(f"\n✓ Saved summary report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs LoRA evaluation results")
    parser.add_argument("--baseline-metrics", type=str,
                       default="baseline_results/baseline_metrics_test.json",
                       help="Path to baseline metrics JSON file")
    parser.add_argument("--lora-metrics", type=str,
                       default="lora_results/lora_metrics_test.json",
                       help="Path to LoRA metrics JSON file")
    parser.add_argument("--baseline-results", type=str,
                       default="baseline_results/baseline_results_test.jsonl",
                       help="Path to baseline results JSONL file")
    parser.add_argument("--lora-results", type=str,
                       default="lora_results/lora_results_test.jsonl",
                       help="Path to LoRA results JSONL file")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                       help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Loading metrics and results...")
    baseline_metrics = load_metrics(args.baseline_metrics)
    lora_metrics = load_metrics(args.lora_metrics)
    baseline_results = load_results(args.baseline_results)
    lora_results = load_results(args.lora_results)
    
    print(f"Baseline: {len(baseline_results)} samples")
    print(f"LoRA: {len(lora_results)} samples")
    
    # Compare metrics
    print("\nComparing metrics...")
    differences = compare_metrics(baseline_metrics, lora_metrics, output_dir)
    
    # Compare predictions
    print("\nComparing individual predictions...")
    comparison_data = compare_predictions(baseline_results, lora_results, output_dir)
    
    # Generate report
    print("\nGenerating summary report...")
    generate_summary_report(baseline_metrics, lora_metrics, differences, 
                          comparison_data, output_dir)
    
    print("\n" + "="*80)
    print("✓✓✓ Comparison complete! ✓✓✓")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()



