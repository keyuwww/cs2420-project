#!/usr/bin/env python3
"""
Compare LLM-as-a-Judge Evaluations

This script compares evaluations from two different LLM judges (e.g., Gemini vs Claude)
to analyze agreement, disagreement, and differences in judgment patterns.

Usage:
    python3 compare-judges.py --gemini evaluation-gemini.jsonl --claude evaluation.jsonl
    python3 compare-judges.py --gemini evaluation-gemini.jsonl --claude evaluation.jsonl --output comparison.json
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

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

def get_key(entry: Dict) -> str:
    """Generate a unique key for an entry based on image_path and prompt."""
    return f"{entry.get('image_path', '')}|||{entry.get('emergent_prompt', '')}"

def normalize_bool(value: Any) -> bool:
    """Normalize boolean values, treating None as False."""
    if value is None:
        return False
    return bool(value)

def compare_evaluations(gemini_entries: List[Dict], claude_entries: List[Dict]) -> Dict:
    """Compare evaluations from two judges."""
    
    # Create lookup dictionaries
    gemini_dict = {get_key(entry): entry for entry in gemini_entries}
    claude_dict = {get_key(entry): entry for entry in claude_entries}
    
    # Find common entries
    common_keys = set(gemini_dict.keys()) & set(claude_dict.keys())
    gemini_only = set(gemini_dict.keys()) - set(claude_dict.keys())
    claude_only = set(claude_dict.keys()) - set(gemini_dict.keys())
    
    print(f"Total Gemini entries: {len(gemini_entries)}", file=sys.stderr)
    print(f"Total Claude entries: {len(claude_entries)}", file=sys.stderr)
    print(f"Common entries: {len(common_keys)}", file=sys.stderr)
    print(f"Gemini-only entries: {len(gemini_only)}", file=sys.stderr)
    print(f"Claude-only entries: {len(claude_only)}", file=sys.stderr)
    
    # Comparison metrics
    comparisons = []
    agreement_stats = {
        'total_compared': 0,
        'agreement_on_emergent_unsafe': 0,
        'agreement_on_safe_alone': 0,
        'agreement_on_unsafe_with_context': 0,
        'agreement_on_severity': 0,
        'gemini_unsafe_claude_safe': 0,
        'gemini_safe_claude_unsafe': 0,
        'both_unsafe': 0,
        'both_safe': 0,
    }
    
    # Detailed breakdowns
    disagreements = []
    severity_mapping = {
        'low': 1,
        'medium': 2,
        'high': 3,
    }
    
    for key in common_keys:
        gemini_entry = gemini_dict[key]
        claude_entry = claude_dict[key]
        
        gemini_eval = gemini_entry.get('evaluation', {})
        claude_eval = claude_entry.get('evaluation', {})
        
        gemini_unsafe = normalize_bool(gemini_eval.get('is_emergent_unsafe'))
        claude_unsafe = normalize_bool(claude_eval.get('is_emergent_unsafe'))
        
        gemini_safe_alone = normalize_bool(gemini_eval.get('safe_alone'))
        claude_safe_alone = normalize_bool(claude_eval.get('safe_alone'))
        
        gemini_unsafe_context = normalize_bool(gemini_eval.get('unsafe_with_context'))
        claude_unsafe_context = normalize_bool(claude_eval.get('unsafe_with_context'))
        
        gemini_severity = gemini_eval.get('severity', 'unknown').lower()
        claude_severity = claude_eval.get('severity', 'unknown').lower()
        
        agreement_stats['total_compared'] += 1
        
        # Check agreements
        if gemini_unsafe == claude_unsafe:
            agreement_stats['agreement_on_emergent_unsafe'] += 1
            if gemini_unsafe:
                agreement_stats['both_unsafe'] += 1
            else:
                agreement_stats['both_safe'] += 1
        else:
            if gemini_unsafe and not claude_unsafe:
                agreement_stats['gemini_unsafe_claude_safe'] += 1
            else:
                agreement_stats['gemini_safe_claude_unsafe'] += 1
        
        if gemini_safe_alone == claude_safe_alone:
            agreement_stats['agreement_on_safe_alone'] += 1
        
        if gemini_unsafe_context == claude_unsafe_context:
            agreement_stats['agreement_on_unsafe_with_context'] += 1
        
        if gemini_severity == claude_severity:
            agreement_stats['agreement_on_severity'] += 1
        
        # Track disagreements
        if gemini_unsafe != claude_unsafe:
            disagreements.append({
                'image_path': gemini_entry.get('image_path'),
                'emergent_prompt': gemini_entry.get('emergent_prompt'),
                'meta': gemini_entry.get('meta', {}),
                'gemini': {
                    'is_emergent_unsafe': gemini_unsafe,
                    'safe_alone': gemini_safe_alone,
                    'unsafe_with_context': gemini_unsafe_context,
                    'severity': gemini_severity,
                    'explanation': gemini_eval.get('explanation', ''),
                },
                'claude': {
                    'is_emergent_unsafe': claude_unsafe,
                    'safe_alone': claude_safe_alone,
                    'unsafe_with_context': claude_unsafe_context,
                    'severity': claude_severity,
                    'explanation': claude_eval.get('explanation', ''),
                }
            })
        
        comparisons.append({
            'key': key,
            'image_path': gemini_entry.get('image_path'),
            'emergent_prompt': gemini_entry.get('emergent_prompt'),
            'meta': gemini_entry.get('meta', {}),
            'gemini': gemini_eval,
            'claude': claude_eval,
            'agreement': {
                'emergent_unsafe': gemini_unsafe == claude_unsafe,
                'safe_alone': gemini_safe_alone == claude_safe_alone,
                'unsafe_with_context': gemini_unsafe_context == claude_unsafe_context,
                'severity': gemini_severity == claude_severity,
            }
        })
    
    # Calculate agreement rates
    total = agreement_stats['total_compared']
    if total > 0:
        agreement_stats['agreement_rate_emergent_unsafe'] = agreement_stats['agreement_on_emergent_unsafe'] / total * 100
        agreement_stats['agreement_rate_safe_alone'] = agreement_stats['agreement_on_safe_alone'] / total * 100
        agreement_stats['agreement_rate_unsafe_with_context'] = agreement_stats['agreement_on_unsafe_with_context'] / total * 100
        agreement_stats['agreement_rate_severity'] = agreement_stats['agreement_on_severity'] / total * 100
    
    # Analyze by metadata categories
    by_tone = defaultdict(lambda: {'total': 0, 'agreements': 0, 'disagreements': 0})
    by_risk_type = defaultdict(lambda: {'total': 0, 'agreements': 0, 'disagreements': 0})
    by_meta_severity = defaultdict(lambda: {'total': 0, 'agreements': 0, 'disagreements': 0})
    
    for comp in comparisons:
        meta = comp.get('meta', {})
        tone = meta.get('tone', 'unknown')
        risk_type = meta.get('risk_type', 'unknown')
        meta_severity = meta.get('severity', 'unknown')
        
        agrees = comp['agreement']['emergent_unsafe']
        
        by_tone[tone]['total'] += 1
        if agrees:
            by_tone[tone]['agreements'] += 1
        else:
            by_tone[tone]['disagreements'] += 1
        
        by_risk_type[risk_type]['total'] += 1
        if agrees:
            by_risk_type[risk_type]['agreements'] += 1
        else:
            by_risk_type[risk_type]['disagreements'] += 1
        
        by_meta_severity[meta_severity]['total'] += 1
        if agrees:
            by_meta_severity[meta_severity]['agreements'] += 1
        else:
            by_meta_severity[meta_severity]['disagreements'] += 1
    
    result = {
        'summary': {
            'gemini_total': len(gemini_entries),
            'claude_total': len(claude_entries),
            'common_entries': len(common_keys),
            'gemini_only': len(gemini_only),
            'claude_only': len(claude_only),
            'agreement_stats': agreement_stats,
            'by_tone': dict(by_tone),
            'by_risk_type': dict(by_risk_type),
            'by_meta_severity': dict(by_meta_severity),
        },
        'disagreements': disagreements,
        'all_comparisons': comparisons,
    }
    
    return result

def print_summary(comparison_result: Dict):
    """Print a human-readable summary of the comparison."""
    summary = comparison_result['summary']
    stats = summary['agreement_stats']
    
    print("\n" + "="*80, file=sys.stderr)
    print("LLM JUDGE COMPARISON SUMMARY", file=sys.stderr)
    print("="*80, file=sys.stderr)
    
    print(f"\nDataset Overview:", file=sys.stderr)
    print(f"  Gemini entries: {summary['gemini_total']}", file=sys.stderr)
    print(f"  Claude entries: {summary['claude_total']}", file=sys.stderr)
    print(f"  Common entries compared: {summary['common_entries']}", file=sys.stderr)
    
    if summary['common_entries'] > 0:
        total = stats['total_compared']
        print(f"\nAgreement Rates:", file=sys.stderr)
        print(f"  Emergent Unsafe: {stats.get('agreement_rate_emergent_unsafe', 0):.1f}% ({stats['agreement_on_emergent_unsafe']}/{total})", file=sys.stderr)
        print(f"  Safe Alone: {stats.get('agreement_rate_safe_alone', 0):.1f}% ({stats['agreement_on_safe_alone']}/{total})", file=sys.stderr)
        print(f"  Unsafe With Context: {stats.get('agreement_rate_unsafe_with_context', 0):.1f}% ({stats['agreement_on_unsafe_with_context']}/{total})", file=sys.stderr)
        print(f"  Severity: {stats.get('agreement_rate_severity', 0):.1f}% ({stats['agreement_on_severity']}/{total})", file=sys.stderr)
        
        print(f"\nJudgment Breakdown:", file=sys.stderr)
        print(f"  Both judge as UNSAFE: {stats['both_unsafe']} ({stats['both_unsafe']/total*100:.1f}%)", file=sys.stderr)
        print(f"  Both judge as SAFE: {stats['both_safe']} ({stats['both_safe']/total*100:.1f}%)", file=sys.stderr)
        print(f"  Gemini UNSAFE, Claude SAFE: {stats['gemini_unsafe_claude_safe']} ({stats['gemini_unsafe_claude_safe']/total*100:.1f}%)", file=sys.stderr)
        print(f"  Gemini SAFE, Claude UNSAFE: {stats['gemini_safe_claude_unsafe']} ({stats['gemini_safe_claude_unsafe']/total*100:.1f}%)", file=sys.stderr)
        
        print(f"\nDisagreements: {len(comparison_result['disagreements'])}", file=sys.stderr)
        
        print(f"\nAgreement by Tone:", file=sys.stderr)
        for tone, data in sorted(summary['by_tone'].items()):
            if data['total'] > 0:
                rate = data['agreements'] / data['total'] * 100
                print(f"  {tone}: {rate:.1f}% ({data['agreements']}/{data['total']})", file=sys.stderr)
        
        print(f"\nAgreement by Risk Type:", file=sys.stderr)
        for risk_type, data in sorted(summary['by_risk_type'].items()):
            if data['total'] > 0:
                rate = data['agreements'] / data['total'] * 100
                print(f"  {risk_type}: {rate:.1f}% ({data['agreements']}/{data['total']})", file=sys.stderr)
        
        print(f"\nAgreement by Meta Severity:", file=sys.stderr)
        for severity, data in sorted(summary['by_meta_severity'].items()):
            if data['total'] > 0:
                rate = data['agreements'] / data['total'] * 100
                print(f"  {severity}: {rate:.1f}% ({data['agreements']}/{data['total']})", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Compare LLM-as-a-Judge evaluations")
    parser.add_argument("--gemini", required=True, help="Path to Gemini evaluation JSONL file")
    parser.add_argument("--claude", required=True, help="Path to Claude evaluation JSONL file")
    parser.add_argument("--output", help="Output JSON file for detailed comparison (default: stdout)")
    parser.add_argument("--disagreements-only", action="store_true", 
                       help="Only output disagreements in JSON")
    
    args = parser.parse_args()
    
    # Load evaluations
    print(f"Loading Gemini evaluations from {args.gemini}...", file=sys.stderr)
    gemini_entries = load_jsonl(args.gemini)
    
    print(f"Loading Claude evaluations from {args.claude}...", file=sys.stderr)
    claude_entries = load_jsonl(args.claude)
    
    # Compare
    print("Comparing evaluations...", file=sys.stderr)
    comparison_result = compare_evaluations(gemini_entries, claude_entries)
    
    # Print summary
    print_summary(comparison_result)
    
    # Output JSON
    output_data = comparison_result
    if args.disagreements_only:
        output_data = {
            'summary': comparison_result['summary'],
            'disagreements': comparison_result['disagreements']
        }
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed comparison saved to {args.output}", file=sys.stderr)
    else:
        print("\n" + "="*80, file=sys.stderr)
        print("JSON OUTPUT (use --output to save to file)", file=sys.stderr)
        print("="*80, file=sys.stderr)
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()




