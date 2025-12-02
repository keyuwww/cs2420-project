#!/usr/bin/env python3
"""
Compare and Visualize Multiple LLM-as-a-Judge Evaluations

This script compares evaluations from multiple LLM judges (Gemini, Updated Gemini, Claude)
and generates comprehensive visualizations.

Usage:
    python3 compare-all-judges.py
    python3 compare-all-judges.py --output-dir plots/
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
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
        return []

def get_key(entry: Dict) -> str:
    """Generate a unique key for an entry."""
    prompt = entry.get('emergent_prompt') or entry.get('emergent_unsafe_prompt', '')
    return f"{entry.get('image_path', '')}|||{prompt}"

def normalize_bool(value: Any) -> bool:
    """Normalize boolean values."""
    if value is None:
        return False
    return bool(value)

def compare_all_judges(gemini_entries, updated_gemini_entries, claude_entries):
    """Compare evaluations from all three judges."""
    
    # Create lookup dictionaries
    gemini_dict = {get_key(entry): entry for entry in gemini_entries}
    updated_gemini_dict = {get_key(entry): entry for entry in updated_gemini_entries}
    claude_dict = {get_key(entry): entry for entry in claude_entries}
    
    # Find common entries
    all_keys = set(gemini_dict.keys()) | set(updated_gemini_dict.keys()) | set(claude_dict.keys())
    common_keys = set(gemini_dict.keys()) & set(updated_gemini_dict.keys()) & set(claude_dict.keys())
    
    print(f"Total entries:", file=sys.stderr)
    print(f"  Gemini: {len(gemini_entries)}", file=sys.stderr)
    print(f"  Updated Gemini: {len(updated_gemini_entries)}", file=sys.stderr)
    print(f"  Claude: {len(claude_entries)}", file=sys.stderr)
    print(f"  Common to all three: {len(common_keys)}", file=sys.stderr)
    
    # Comparison data
    comparisons = []
    for key in common_keys:
        gemini_entry = gemini_dict[key]
        updated_gemini_entry = updated_gemini_dict[key]
        claude_entry = claude_dict[key]
        
        gemini_eval = gemini_entry.get('evaluation', {})
        updated_gemini_eval = updated_gemini_entry.get('evaluation', {})
        claude_eval = claude_entry.get('evaluation', {})
        
        gemini_unsafe = normalize_bool(gemini_eval.get('is_emergent_unsafe'))
        updated_gemini_unsafe = normalize_bool(updated_gemini_eval.get('is_emergent_unsafe'))
        claude_unsafe = normalize_bool(claude_eval.get('is_emergent_unsafe'))
        
        def get_severity(eval_dict):
            sev = eval_dict.get('severity')
            if sev is None:
                return 'unknown'
            return str(sev).lower()
        
        comparisons.append({
            'key': key,
            'image_path': gemini_entry.get('image_path'),
            'emergent_prompt': gemini_entry.get('emergent_prompt') or gemini_entry.get('emergent_unsafe_prompt', ''),
            'meta': gemini_entry.get('meta', {}),
            'gemini': {
                'is_emergent_unsafe': gemini_unsafe,
                'safe_alone': normalize_bool(gemini_eval.get('safe_alone')),
                'unsafe_with_context': normalize_bool(gemini_eval.get('unsafe_with_context')),
                'severity': get_severity(gemini_eval),
            },
            'updated_gemini': {
                'is_emergent_unsafe': updated_gemini_unsafe,
                'safe_alone': normalize_bool(updated_gemini_eval.get('safe_alone')),
                'unsafe_with_context': normalize_bool(updated_gemini_eval.get('unsafe_with_context')),
                'severity': get_severity(updated_gemini_eval),
            },
            'claude': {
                'is_emergent_unsafe': claude_unsafe,
                'safe_alone': normalize_bool(claude_eval.get('safe_alone')),
                'unsafe_with_context': normalize_bool(claude_eval.get('unsafe_with_context')),
                'severity': get_severity(claude_eval),
            }
        })
    
    return comparisons, common_keys

def plot_three_way_agreement_matrix(comparisons, output_dir):
    """Plot three-way agreement matrix."""
    # Count agreement patterns
    patterns = defaultdict(int)
    
    for comp in comparisons:
        g = comp['gemini']['is_emergent_unsafe']
        ug = comp['updated_gemini']['is_emergent_unsafe']
        c = comp['claude']['is_emergent_unsafe']
        
        pattern = f"{'U' if g else 'S'}{'U' if ug else 'S'}{'U' if c else 'S'}"
        patterns[pattern] += 1
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    pattern_labels = {
        'SSS': 'All Safe',
        'SSU': 'Gemini+Updated Safe, Claude Unsafe',
        'SUS': 'Gemini+Claude Safe, Updated Unsafe',
        'SUU': 'Gemini Safe, Others Unsafe',
        'USS': 'Gemini Unsafe, Others Safe',
        'USU': 'Gemini+Claude Unsafe, Updated Safe',
        'UUS': 'Gemini+Updated Unsafe, Claude Safe',
        'UUU': 'All Unsafe',
    }
    
    sorted_patterns = sorted(patterns.items())
    labels = [pattern_labels.get(p, p) for p, _ in sorted_patterns]
    values = [v for _, v in sorted_patterns]
    colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
    
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title('Three-Way Agreement Patterns\n(Gemini vs Updated Gemini vs Claude)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{val} ({val/sum(values)*100:.1f}%)',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'three_way_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'three_way_agreement.png'}", file=sys.stderr)

def plot_pairwise_agreement_matrices(comparisons, output_dir):
    """Plot pairwise agreement matrices for each pair."""
    pairs = [
        ('gemini', 'updated_gemini', 'Gemini', 'Updated Gemini'),
        ('gemini', 'claude', 'Gemini', 'Claude'),
        ('updated_gemini', 'claude', 'Updated Gemini', 'Claude'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (judge1, judge2, label1, label2) in enumerate(pairs):
        ax = axes[idx]
        
        matrix = {
            ('UNSAFE', 'UNSAFE'): 0,
            ('UNSAFE', 'SAFE'): 0,
            ('SAFE', 'UNSAFE'): 0,
            ('SAFE', 'SAFE'): 0,
        }
        
        for comp in comparisons:
            j1_unsafe = comp[judge1]['is_emergent_unsafe']
            j2_unsafe = comp[judge2]['is_emergent_unsafe']
            
            j1_label = 'UNSAFE' if j1_unsafe else 'SAFE'
            j2_label = 'UNSAFE' if j2_unsafe else 'SAFE'
            
            matrix[(j1_label, j2_label)] += 1
        
        data = np.array([
            [matrix[('UNSAFE', 'UNSAFE')], matrix[('UNSAFE', 'SAFE')]],
            [matrix[('SAFE', 'UNSAFE')], matrix[('SAFE', 'SAFE')]]
        ])
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=data.max())
        
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, f'{data[i, j]}\n({data[i, j]/data.sum()*100:.1f}%)',
                              ha="center", va="center", color="black", fontsize=11, fontweight='bold')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f'{label2}: UNSAFE', f'{label2}: SAFE'])
        ax.set_yticklabels([f'{label1}: UNSAFE', f'{label1}: SAFE'])
        ax.set_title(f'{label1} vs {label2}', fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Count')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pairwise_agreement_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'pairwise_agreement_matrices.png'}", file=sys.stderr)

def plot_severity_comparison(comparisons, output_dir):
    """Plot severity distribution comparison."""
    severities = {
        'gemini': [],
        'updated_gemini': [],
        'claude': []
    }
    
    for comp in comparisons:
        severities['gemini'].append(comp['gemini']['severity'])
        severities['updated_gemini'].append(comp['updated_gemini']['severity'])
        severities['claude'].append(comp['claude']['severity'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (judge, label) in enumerate([('gemini', 'Gemini'), 
                                          ('updated_gemini', 'Updated Gemini'), 
                                          ('claude', 'Claude')]):
        ax = axes[idx]
        counts = Counter(severities[judge])
        
        bars = ax.bar(counts.keys(), counts.values(), color=['#3498DB', '#E67E22', '#27AE60', '#95A5A6'][:len(counts)])
        ax.set_title(f'{label} Severity Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Severity')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        
        total = sum(counts.values())
        for k, v in counts.items():
            ax.text(k, v, f'{v}\n({v/total*100:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'severity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'severity_comparison.png'}", file=sys.stderr)

def plot_judgment_distribution(comparisons, output_dir):
    """Plot judgment distribution for each judge."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    judges = ['gemini', 'updated_gemini', 'claude']
    labels = ['Gemini', 'Updated Gemini', 'Claude']
    
    unsafe_counts = []
    safe_counts = []
    
    for judge in judges:
        unsafe = sum(1 for comp in comparisons if comp[judge]['is_emergent_unsafe'])
        safe = len(comparisons) - unsafe
        unsafe_counts.append(unsafe)
        safe_counts.append(safe)
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, unsafe_counts, width, label='UNSAFE', color='#E74C3C', alpha=0.7)
    bars2 = ax.bar(x + width/2, safe_counts, width, label='SAFE', color='#95A5A6', alpha=0.7)
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Judgment Distribution by Judge', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(comparisons)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'judgment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'judgment_distribution.png'}", file=sys.stderr)

def plot_agreement_by_category(comparisons, output_dir):
    """Plot agreement rates by metadata categories."""
    by_tone = defaultdict(lambda: {'total': 0, 'all_agree': 0, 'pairwise_agree': defaultdict(int)})
    by_risk_type = defaultdict(lambda: {'total': 0, 'all_agree': 0, 'pairwise_agree': defaultdict(int)})
    
    for comp in comparisons:
        meta = comp.get('meta', {})
        tone = meta.get('tone', 'unknown')
        risk_type = meta.get('risk_type', 'unknown')
        
        g = comp['gemini']['is_emergent_unsafe']
        ug = comp['updated_gemini']['is_emergent_unsafe']
        c = comp['claude']['is_emergent_unsafe']
        
        all_agree = (g == ug == c)
        
        by_tone[tone]['total'] += 1
        if all_agree:
            by_tone[tone]['all_agree'] += 1
        
        by_risk_type[risk_type]['total'] += 1
        if all_agree:
            by_risk_type[risk_type]['all_agree'] += 1
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # By tone
    tones = sorted(by_tone.keys())
    tone_rates = [by_tone[t]['all_agree'] / by_tone[t]['total'] * 100 if by_tone[t]['total'] > 0 else 0 
                  for t in tones]
    axes[0].bar(tones, tone_rates, color='#9B59B6', alpha=0.7)
    axes[0].set_title('Three-Way Agreement Rate by Tone', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tone')
    axes[0].set_ylabel('Agreement Rate (%)')
    axes[0].set_ylim([0, 100])
    axes[0].tick_params(axis='x', rotation=45)
    for i, (t, rate) in enumerate(zip(tones, tone_rates)):
        axes[0].text(i, rate, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # By risk type
    risk_types = sorted(by_risk_type.keys())
    risk_rates = [by_risk_type[r]['all_agree'] / by_risk_type[r]['total'] * 100 if by_risk_type[r]['total'] > 0 else 0 
                  for r in risk_types]
    axes[1].bar(risk_types, risk_rates, color='#E74C3C', alpha=0.7)
    axes[1].set_title('Three-Way Agreement Rate by Risk Type', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Risk Type')
    axes[1].set_ylabel('Agreement Rate (%)')
    axes[1].set_ylim([0, 100])
    axes[1].tick_params(axis='x', rotation=45)
    for i, (r, rate) in enumerate(zip(risk_types, risk_rates)):
        axes[1].text(i, rate, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'agreement_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'agreement_by_category.png'}", file=sys.stderr)

def plot_component_agreement(comparisons, output_dir):
    """Plot agreement on individual components."""
    components = ['safe_alone', 'unsafe_with_context', 'is_emergent_unsafe']
    component_labels = ['Safe Alone', 'Unsafe With Context', 'Overall (Emergent Unsafe)']
    
    agreement_rates = {
        'gemini_vs_updated': [],
        'gemini_vs_claude': [],
        'updated_vs_claude': []
    }
    
    for comp_name in components:
        # Gemini vs Updated Gemini
        agree = sum(1 for c in comparisons 
                   if c['gemini'][comp_name] == c['updated_gemini'][comp_name])
        agreement_rates['gemini_vs_updated'].append(agree / len(comparisons) * 100)
        
        # Gemini vs Claude
        agree = sum(1 for c in comparisons 
                   if c['gemini'][comp_name] == c['claude'][comp_name])
        agreement_rates['gemini_vs_claude'].append(agree / len(comparisons) * 100)
        
        # Updated Gemini vs Claude
        agree = sum(1 for c in comparisons 
                   if c['updated_gemini'][comp_name] == c['claude'][comp_name])
        agreement_rates['updated_vs_claude'].append(agree / len(comparisons) * 100)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(component_labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, agreement_rates['gemini_vs_updated'], width, 
                   label='Gemini vs Updated Gemini', color='#3498DB', alpha=0.7)
    bars2 = ax.bar(x, agreement_rates['gemini_vs_claude'], width,
                   label='Gemini vs Claude', color='#E67E22', alpha=0.7)
    bars3 = ax.bar(x + width, agreement_rates['updated_vs_claude'], width,
                   label='Updated Gemini vs Claude', color='#27AE60', alpha=0.7)
    
    ax.set_ylabel('Agreement Rate (%)', fontsize=12)
    ax.set_title('Pairwise Agreement Rate by Evaluation Component', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(component_labels)
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'component_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'component_agreement.png'}", file=sys.stderr)

def print_summary(comparisons):
    """Print summary statistics."""
    total = len(comparisons)
    
    # Three-way agreement
    all_agree = sum(1 for c in comparisons 
                   if (c['gemini']['is_emergent_unsafe'] == 
                       c['updated_gemini']['is_emergent_unsafe'] == 
                       c['claude']['is_emergent_unsafe']))
    
    # Pairwise agreements
    gemini_updated_agree = sum(1 for c in comparisons 
                              if c['gemini']['is_emergent_unsafe'] == c['updated_gemini']['is_emergent_unsafe'])
    gemini_claude_agree = sum(1 for c in comparisons 
                             if c['gemini']['is_emergent_unsafe'] == c['claude']['is_emergent_unsafe'])
    updated_claude_agree = sum(1 for c in comparisons 
                              if c['updated_gemini']['is_emergent_unsafe'] == c['claude']['is_emergent_unsafe'])
    
    print("\n" + "="*80, file=sys.stderr)
    print("COMPARISON SUMMARY", file=sys.stderr)
    print("="*80, file=sys.stderr)
    print(f"\nTotal entries compared: {total}", file=sys.stderr)
    print(f"\nThree-Way Agreement (all judges agree): {all_agree} ({all_agree/total*100:.1f}%)", file=sys.stderr)
    print(f"\nPairwise Agreements:", file=sys.stderr)
    print(f"  Gemini vs Updated Gemini: {gemini_updated_agree} ({gemini_updated_agree/total*100:.1f}%)", file=sys.stderr)
    print(f"  Gemini vs Claude: {gemini_claude_agree} ({gemini_claude_agree/total*100:.1f}%)", file=sys.stderr)
    print(f"  Updated Gemini vs Claude: {updated_claude_agree} ({updated_claude_agree/total*100:.1f}%)", file=sys.stderr)
    
    # Judgment counts
    print(f"\nJudgment Counts:", file=sys.stderr)
    for judge, label in [('gemini', 'Gemini'), ('updated_gemini', 'Updated Gemini'), ('claude', 'Claude')]:
        unsafe = sum(1 for c in comparisons if c[judge]['is_emergent_unsafe'])
        safe = total - unsafe
        print(f"  {label}: {unsafe} UNSAFE ({unsafe/total*100:.1f}%), {safe} SAFE ({safe/total*100:.1f}%)", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Compare and visualize multiple LLM judge evaluations")
    parser.add_argument("--gemini", default="image-prompt/processed_results/evaluation-gemini.jsonl",
                       help="Path to Gemini evaluation JSONL file")
    parser.add_argument("--updated-gemini", default="image-prompt/processed_results/evaluation-updated-gemini.jsonl",
                       help="Path to Updated Gemini evaluation JSONL file")
    parser.add_argument("--claude", default="image-prompt/processed_results/evaluation-claude.jsonl",
                       help="Path to Claude evaluation JSONL file")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots (default: plots/)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load evaluations
    print(f"Loading Gemini evaluations from {args.gemini}...", file=sys.stderr)
    gemini_entries = load_jsonl(args.gemini)
    
    print(f"Loading Updated Gemini evaluations from {args.updated_gemini}...", file=sys.stderr)
    updated_gemini_entries = load_jsonl(args.updated_gemini)
    
    print(f"Loading Claude evaluations from {args.claude}...", file=sys.stderr)
    claude_entries = load_jsonl(args.claude)
    
    # Compare
    print("Comparing evaluations...", file=sys.stderr)
    comparisons, common_keys = compare_all_judges(gemini_entries, updated_gemini_entries, claude_entries)
    
    if len(comparisons) == 0:
        print("ERROR: No common entries found between all three files!", file=sys.stderr)
        sys.exit(1)
    
    # Print summary
    print_summary(comparisons)
    
    # Generate plots
    print(f"\nGenerating plots in {output_dir}...", file=sys.stderr)
    plot_three_way_agreement_matrix(comparisons, output_dir)
    plot_pairwise_agreement_matrices(comparisons, output_dir)
    plot_severity_comparison(comparisons, output_dir)
    plot_judgment_distribution(comparisons, output_dir)
    plot_agreement_by_category(comparisons, output_dir)
    plot_component_agreement(comparisons, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/", file=sys.stderr)

if __name__ == "__main__":
    main()

