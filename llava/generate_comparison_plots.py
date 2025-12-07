#!/usr/bin/env python3
"""
Generate comprehensive comparison plots for Baseline vs LoRA results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_metrics(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_results(filepath):
    """Load detailed results from JSONL file"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def plot_metrics_comparison(baseline_metrics, lora_metrics, output_dir):
    """Plot side-by-side metrics comparison"""
    metrics = ['accuracy', 'auroc', 'precision', 'recall', 'f1']
    baseline_vals = [baseline_metrics.get(m, 0) for m in metrics]
    lora_vals = [lora_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, lora_vals, width, label='LoRA', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Baseline vs LoRA: Metrics Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics_comparison.png")

def plot_confusion_matrices(baseline_metrics, lora_metrics, output_dir):
    """Plot confusion matrices side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline confusion matrix
    cm_baseline = np.array(baseline_metrics['confusion_matrix'])
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['YES (Safe)', 'NO (Unsafe)'],
                yticklabels=['Safe', 'Unsafe'],
                cbar_kws={'label': 'Count'})
    ax1.set_title('Baseline Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # LoRA confusion matrix
    cm_lora = np.array(lora_metrics['confusion_matrix'])
    sns.heatmap(cm_lora, annot=True, fmt='d', cmap='Reds', ax=ax2,
                xticklabels=['YES (Safe)', 'NO (Unsafe)'],
                yticklabels=['Safe', 'Unsafe'],
                cbar_kws={'label': 'Count'})
    ax2.set_title('LoRA Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion_matrices.png")

def plot_improvement_metrics(baseline_metrics, lora_metrics, output_dir):
    """Plot improvement percentages"""
    metrics = ['accuracy', 'auroc', 'precision', 'recall', 'f1']
    improvements = []
    for m in metrics:
        baseline_val = baseline_metrics.get(m, 0)
        lora_val = lora_metrics.get(m, 0)
        improvement = ((lora_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
        improvements.append(improvement)
    
    colors = ['green' if x > 0 else 'red' for x in improvements]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh([m.replace('_', ' ').title() for m in metrics], improvements, color=colors, alpha=0.7)
    
    ax.set_xlabel('Improvement (%)', fontsize=14)
    ax.set_title('LoRA Improvement Over Baseline', fontsize=16, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        ax.text(val, i, f'{val:+.1f}%',
               ha='left' if val > 0 else 'right', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved improvement_metrics.png")

def plot_p_emergent_distribution(lora_results, output_dir):
    """Plot distribution of P_emergent scores"""
    safe_p_emergent = [r['p_emergent'] for r in lora_results if r['label'] == 0]
    unsafe_p_emergent = [r['p_emergent'] for r in lora_results if r['label'] == 1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(safe_p_emergent, bins=30, alpha=0.6, label='Safe (Label=0)', color='green', edgecolor='black')
    ax.hist(unsafe_p_emergent, bins=30, alpha=0.6, label='Unsafe (Label=1)', color='red', edgecolor='black')
    
    # Add threshold line
    threshold = lora_results[0].get('threshold', 0.5)
    ax.axvline(x=threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
    
    ax.set_xlabel('P_emergent (Probability)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Distribution of P_emergent Scores by True Label', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'p_emergent_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved p_emergent_distribution.png")

def plot_prediction_agreement(baseline_results, lora_results, output_dir):
    """Plot where baseline and LoRA agree/disagree"""
    if len(baseline_results) != len(lora_results):
        print("Warning: Different number of samples, skipping agreement plot")
        return
    
    agreements = {'both_correct': 0, 'both_wrong': 0, 'baseline_correct': 0, 'lora_correct': 0}
    
    for b, l in zip(baseline_results, lora_results):
        b_correct = b.get('correct', 0)
        l_correct = l.get('correct', 0)
        
        if b_correct and l_correct:
            agreements['both_correct'] += 1
        elif not b_correct and not l_correct:
            agreements['both_wrong'] += 1
        elif b_correct and not l_correct:
            agreements['baseline_correct'] += 1
        elif not b_correct and l_correct:
            agreements['lora_correct'] += 1
    
    labels = list(agreements.keys())
    values = list(agreements.values())
    colors = ['green', 'red', 'orange', 'blue']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Number of Samples', fontsize=14)
    ax.set_title('Prediction Agreement: Baseline vs LoRA', fontsize=16, fontweight='bold')
    ax.set_xticklabels([l.replace('_', ' ').title() for l in labels], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved prediction_agreement.png")

def plot_class_accuracy_comparison(baseline_metrics, lora_metrics, output_dir):
    """Plot accuracy by class (safe vs unsafe)"""
    classes = ['Safe', 'Unsafe']
    baseline_acc = [baseline_metrics.get('safe_accuracy', 0), baseline_metrics.get('unsafe_accuracy', 0)]
    lora_acc = [lora_metrics.get('safe_accuracy', 0), lora_metrics.get('unsafe_accuracy', 0)]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_acc, width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, lora_acc, width, label='LoRA', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Accuracy by Class: Baseline vs LoRA', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved class_accuracy_comparison.png")

def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots")
    parser.add_argument("--baseline-metrics", type=str,
                       default="baseline_results/baseline_metrics_test.json",
                       help="Path to baseline metrics")
    parser.add_argument("--lora-metrics", type=str,
                       default="lora_results/lora_metrics_test.json",
                       help="Path to LoRA metrics")
    parser.add_argument("--baseline-results", type=str,
                       default="baseline_results/baseline_results_test.jsonl",
                       help="Path to baseline detailed results")
    parser.add_argument("--lora-results", type=str,
                       default="lora_results/lora_results_test.jsonl",
                       help="Path to LoRA detailed results")
    parser.add_argument("--output-dir", type=str, default="comparison_plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Load metrics
    print("Loading metrics...")
    baseline_metrics = load_metrics(args.baseline_metrics)
    lora_metrics = load_metrics(args.lora_metrics)
    print("✓ Metrics loaded")
    print()
    
    # Load detailed results
    print("Loading detailed results...")
    baseline_results = load_results(args.baseline_results)
    lora_results = load_results(args.lora_results)
    print(f"✓ Loaded {len(baseline_results)} baseline results")
    print(f"✓ Loaded {len(lora_results)} LoRA results")
    print()
    
    # Generate plots
    print("Generating plots...")
    print()
    
    plot_metrics_comparison(baseline_metrics, lora_metrics, output_dir)
    plot_confusion_matrices(baseline_metrics, lora_metrics, output_dir)
    plot_improvement_metrics(baseline_metrics, lora_metrics, output_dir)
    plot_class_accuracy_comparison(baseline_metrics, lora_metrics, output_dir)
    plot_p_emergent_distribution(lora_results, output_dir)
    plot_prediction_agreement(baseline_results, lora_results, output_dir)
    
    print()
    print("="*60)
    print("All plots generated successfully!")
    print(f"Plots saved to: {output_dir.absolute()}")
    print("="*60)

if __name__ == "__main__":
    main()


