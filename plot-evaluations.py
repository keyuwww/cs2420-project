#!/usr/bin/env python3
"""
Generate Plots for LLM-as-a-Judge Evaluation Results

This script generates various visualizations to analyze evaluation results,
including agreement rates, severity distributions, and breakdowns by metadata.

Usage:
    python3 plot-evaluations.py --gemini evaluation-gemini.jsonl --claude evaluation.jsonl
    python3 plot-evaluations.py --gemini evaluation-gemini.jsonl --claude evaluation.jsonl --output-dir plots/
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    entries = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries
    except Exception as e:
        print(f"Error loading {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

def normalize_bool(value) -> bool:
    """Normalize boolean values."""
    if value is None:
        return False
    return bool(value)

def get_key(entry: Dict) -> str:
    """Generate a unique key for an entry."""
    return f"{entry.get('image_path', '')}|||{entry.get('emergent_prompt', '')}"

def plot_agreement_matrix(gemini_entries: List[Dict], claude_entries: List[Dict], output_dir: Path):
    """Plot agreement/disagreement matrix."""
    gemini_dict = {get_key(entry): entry for entry in gemini_entries}
    claude_dict = {get_key(entry): entry for entry in claude_entries}
    common_keys = set(gemini_dict.keys()) & set(claude_dict.keys())
    
    # Count agreement patterns
    matrix = {
        ('UNSAFE', 'UNSAFE'): 0,
        ('UNSAFE', 'SAFE'): 0,
        ('SAFE', 'UNSAFE'): 0,
        ('SAFE', 'SAFE'): 0,
    }
    
    for key in common_keys:
        gemini_unsafe = normalize_bool(gemini_dict[key]['evaluation'].get('is_emergent_unsafe'))
        claude_unsafe = normalize_bool(claude_dict[key]['evaluation'].get('is_emergent_unsafe'))
        
        gemini_label = 'UNSAFE' if gemini_unsafe else 'SAFE'
        claude_label = 'UNSAFE' if claude_unsafe else 'SAFE'
        
        matrix[(gemini_label, claude_label)] += 1
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = np.array([
        [matrix[('UNSAFE', 'UNSAFE')], matrix[('UNSAFE', 'SAFE')]],
        [matrix[('SAFE', 'UNSAFE')], matrix[('SAFE', 'SAFE')]]
    ])
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=data.max())
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{data[i, j]}\n({data[i, j]/data.sum()*100:.1f}%)',
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Claude: UNSAFE', 'Claude: SAFE'])
    ax.set_yticklabels(['Gemini: UNSAFE', 'Gemini: SAFE'])
    ax.set_title('Agreement Matrix: Gemini vs Claude\n(Emergent Unsafe Judgments)', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(output_dir / 'agreement_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'agreement_matrix.png'}", file=sys.stderr)

def plot_severity_comparison(gemini_entries: List[Dict], claude_entries: List[Dict], output_dir: Path):
    """Plot severity distribution comparison."""
    gemini_dict = {get_key(entry): entry for entry in gemini_entries}
    claude_dict = {get_key(entry): entry for entry in claude_entries}
    common_keys = set(gemini_dict.keys()) & set(claude_dict.keys())
    
    gemini_severities = []
    claude_severities = []
    
    for key in common_keys:
        gemini_sev = gemini_dict[key]['evaluation'].get('severity', 'unknown').lower()
        claude_sev = claude_dict[key]['evaluation'].get('severity', 'unknown').lower()
        gemini_severities.append(gemini_sev)
        claude_severities.append(claude_sev)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gemini severity distribution
    gemini_counts = Counter(gemini_severities)
    ax1.bar(gemini_counts.keys(), gemini_counts.values(), color='#4285F4', alpha=0.7)
    ax1.set_title('Gemini Severity Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Severity')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add percentages
    total = sum(gemini_counts.values())
    for k, v in gemini_counts.items():
        ax1.text(k, v, f'{v}\n({v/total*100:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # Claude severity distribution
    claude_counts = Counter(claude_severities)
    ax2.bar(claude_counts.keys(), claude_counts.values(), color='#FF6B35', alpha=0.7)
    ax2.set_title('Claude Severity Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Severity')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add percentages
    total = sum(claude_counts.values())
    for k, v in claude_counts.items():
        ax2.text(k, v, f'{v}\n({v/total*100:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'severity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'severity_comparison.png'}", file=sys.stderr)

def plot_agreement_by_category(gemini_entries: List[Dict], claude_entries: List[Dict], output_dir: Path):
    """Plot agreement rates by metadata categories."""
    gemini_dict = {get_key(entry): entry for entry in gemini_entries}
    claude_dict = {get_key(entry): entry for entry in claude_entries}
    common_keys = set(gemini_dict.keys()) & set(claude_dict.keys())
    
    # Group by tone, risk_type, and meta severity
    by_tone = defaultdict(lambda: {'total': 0, 'agreements': 0})
    by_risk_type = defaultdict(lambda: {'total': 0, 'agreements': 0})
    by_severity = defaultdict(lambda: {'total': 0, 'agreements': 0})
    
    for key in common_keys:
        gemini_entry = gemini_dict[key]
        claude_entry = claude_dict[key]
        
        meta = gemini_entry.get('meta', {})
        tone = meta.get('tone', 'unknown')
        risk_type = meta.get('risk_type', 'unknown')
        severity = meta.get('severity', 'unknown')
        
        gemini_unsafe = normalize_bool(gemini_entry['evaluation'].get('is_emergent_unsafe'))
        claude_unsafe = normalize_bool(claude_entry['evaluation'].get('is_emergent_unsafe'))
        agrees = gemini_unsafe == claude_unsafe
        
        by_tone[tone]['total'] += 1
        if agrees:
            by_tone[tone]['agreements'] += 1
        
        by_risk_type[risk_type]['total'] += 1
        if agrees:
            by_risk_type[risk_type]['agreements'] += 1
        
        by_severity[severity]['total'] += 1
        if agrees:
            by_severity[severity]['agreements'] += 1
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # By tone
    tones = sorted(by_tone.keys())
    tone_rates = [by_tone[t]['agreements'] / by_tone[t]['total'] * 100 if by_tone[t]['total'] > 0 else 0 
                  for t in tones]
    axes[0].bar(tones, tone_rates, color='#9B59B6', alpha=0.7)
    axes[0].set_title('Agreement Rate by Tone', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tone')
    axes[0].set_ylabel('Agreement Rate (%)')
    axes[0].set_ylim([0, 100])
    axes[0].tick_params(axis='x', rotation=45)
    for i, (t, rate) in enumerate(zip(tones, tone_rates)):
        axes[0].text(i, rate, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # By risk type
    risk_types = sorted(by_risk_type.keys())
    risk_rates = [by_risk_type[r]['agreements'] / by_risk_type[r]['total'] * 100 if by_risk_type[r]['total'] > 0 else 0 
                  for r in risk_types]
    axes[1].bar(risk_types, risk_rates, color='#E74C3C', alpha=0.7)
    axes[1].set_title('Agreement Rate by Risk Type', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Risk Type')
    axes[1].set_ylabel('Agreement Rate (%)')
    axes[1].set_ylim([0, 100])
    axes[1].tick_params(axis='x', rotation=45)
    for i, (r, rate) in enumerate(zip(risk_types, risk_rates)):
        axes[1].text(i, rate, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # By severity
    severities = sorted(by_severity.keys())
    sev_rates = [by_severity[s]['agreements'] / by_severity[s]['total'] * 100 if by_severity[s]['total'] > 0 else 0 
                 for s in severities]
    axes[2].bar(severities, sev_rates, color='#F39C12', alpha=0.7)
    axes[2].set_title('Agreement Rate by Meta Severity', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Severity')
    axes[2].set_ylabel('Agreement Rate (%)')
    axes[2].set_ylim([0, 100])
    axes[2].tick_params(axis='x', rotation=45)
    for i, (s, rate) in enumerate(zip(severities, sev_rates)):
        axes[2].text(i, rate, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'agreement_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'agreement_by_category.png'}", file=sys.stderr)

def plot_component_agreement(gemini_entries: List[Dict], claude_entries: List[Dict], output_dir: Path):
    """Plot agreement on individual components (safe_alone, unsafe_with_context)."""
    gemini_dict = {get_key(entry): entry for entry in gemini_entries}
    claude_dict = {get_key(entry): entry for entry in claude_entries}
    common_keys = set(gemini_dict.keys()) & set(claude_dict.keys())
    
    safe_alone_agreements = 0
    unsafe_context_agreements = 0
    total = len(common_keys)
    
    for key in common_keys:
        gemini_eval = gemini_dict[key]['evaluation']
        claude_eval = claude_dict[key]['evaluation']
        
        if normalize_bool(gemini_eval.get('safe_alone')) == normalize_bool(claude_eval.get('safe_alone')):
            safe_alone_agreements += 1
        
        if normalize_bool(gemini_eval.get('unsafe_with_context')) == normalize_bool(claude_eval.get('unsafe_with_context')):
            unsafe_context_agreements += 1
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = ['Safe Alone', 'Unsafe With Context', 'Overall (Emergent Unsafe)']
    agreement_rates = [
        safe_alone_agreements / total * 100 if total > 0 else 0,
        unsafe_context_agreements / total * 100 if total > 0 else 0,
    ]
    
    # Calculate overall agreement
    overall_agreements = 0
    for key in common_keys:
        gemini_unsafe = normalize_bool(gemini_dict[key]['evaluation'].get('is_emergent_unsafe'))
        claude_unsafe = normalize_bool(claude_dict[key]['evaluation'].get('is_emergent_unsafe'))
        if gemini_unsafe == claude_unsafe:
            overall_agreements += 1
    agreement_rates.append(overall_agreements / total * 100 if total > 0 else 0)
    
    bars = ax.bar(components, agreement_rates, color=['#3498DB', '#E67E22', '#27AE60'], alpha=0.7)
    ax.set_title('Agreement Rate by Evaluation Component', fontsize=14, fontweight='bold')
    ax.set_ylabel('Agreement Rate (%)')
    ax.set_ylim([0, 100])
    
    for bar, rate in zip(bars, agreement_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'component_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'component_agreement.png'}", file=sys.stderr)

def plot_judgment_distribution(gemini_entries: List[Dict], claude_entries: List[Dict], output_dir: Path):
    """Plot distribution of judgments from each model."""
    gemini_unsafe_count = sum(1 for e in gemini_entries 
                             if normalize_bool(e.get('evaluation', {}).get('is_emergent_unsafe')))
    gemini_safe_count = len(gemini_entries) - gemini_unsafe_count
    
    claude_unsafe_count = sum(1 for e in claude_entries 
                             if normalize_bool(e.get('evaluation', {}).get('is_emergent_unsafe')))
    claude_safe_count = len(claude_entries) - claude_unsafe_count
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gemini distribution
    labels = ['SAFE', 'UNSAFE']
    gemini_sizes = [gemini_safe_count, gemini_unsafe_count]
    colors = ['#95A5A6', '#E74C3C']
    
    ax1.pie(gemini_sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Gemini Judgment Distribution', fontsize=12, fontweight='bold')
    
    # Claude distribution
    claude_sizes = [claude_safe_count, claude_unsafe_count]
    ax2.pie(claude_sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Claude Judgment Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'judgment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'judgment_distribution.png'}", file=sys.stderr)

def plot_severity_heatmap(gemini_entries: List[Dict], claude_entries: List[Dict], output_dir: Path):
    """Plot severity agreement heatmap."""
    gemini_dict = {get_key(entry): entry for entry in gemini_entries}
    claude_dict = {get_key(entry): entry for entry in claude_entries}
    common_keys = set(gemini_dict.keys()) & set(claude_dict.keys())
    
    severity_counts = defaultdict(lambda: defaultdict(int))
    all_severities = set()
    
    for key in common_keys:
        gemini_sev = gemini_dict[key]['evaluation'].get('severity', 'unknown').lower()
        claude_sev = claude_dict[key]['evaluation'].get('severity', 'unknown').lower()
        severity_counts[gemini_sev][claude_sev] += 1
        all_severities.add(gemini_sev)
        all_severities.add(claude_sev)
    
    all_severities = sorted(all_severities)
    if 'unknown' in all_severities:
        all_severities.remove('unknown')
        all_severities.append('unknown')
    
    # Create matrix
    matrix = np.zeros((len(all_severities), len(all_severities)))
    for i, g_sev in enumerate(all_severities):
        for j, c_sev in enumerate(all_severities):
            matrix[i, j] = severity_counts[g_sev][c_sev]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Add text annotations
    for i in range(len(all_severities)):
        for j in range(len(all_severities)):
            text = ax.text(j, i, int(matrix[i, j]),
                          ha="center", va="center", color="black", fontsize=10)
    
    ax.set_xticks(range(len(all_severities)))
    ax.set_yticks(range(len(all_severities)))
    ax.set_xticklabels(all_severities, rotation=45, ha='right')
    ax.set_yticklabels(all_severities)
    ax.set_xlabel('Claude Severity', fontsize=12)
    ax.set_ylabel('Gemini Severity', fontsize=12)
    ax.set_title('Severity Agreement Heatmap', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(output_dir / 'severity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'severity_heatmap.png'}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Generate plots for LLM evaluation comparisons")
    parser.add_argument("--gemini", required=True, help="Path to Gemini evaluation JSONL file")
    parser.add_argument("--claude", required=True, help="Path to Claude evaluation JSONL file")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots (default: plots/)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluations
    print(f"Loading Gemini evaluations from {args.gemini}...", file=sys.stderr)
    gemini_entries = load_jsonl(args.gemini)
    
    print(f"Loading Claude evaluations from {args.claude}...", file=sys.stderr)
    claude_entries = load_jsonl(args.claude)
    
    print(f"Generating plots in {output_dir}...", file=sys.stderr)
    
    # Generate all plots
    plot_agreement_matrix(gemini_entries, claude_entries, output_dir)
    plot_severity_comparison(gemini_entries, claude_entries, output_dir)
    plot_agreement_by_category(gemini_entries, claude_entries, output_dir)
    plot_component_agreement(gemini_entries, claude_entries, output_dir)
    plot_judgment_distribution(gemini_entries, claude_entries, output_dir)
    plot_severity_heatmap(gemini_entries, claude_entries, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/", file=sys.stderr)

if __name__ == "__main__":
    main()




