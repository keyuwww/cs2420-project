#!/usr/bin/env python3
"""
Analyze voting results and compare against LLM judge evaluations.
Uses a weighted system: at least 3 "yes" votes = "yes" as gold truth label.
"""

import json
from collections import defaultdict
from pathlib import Path

def load_voting_files(voting_files):
    """Load all voting result files and aggregate votes."""
    votes = defaultdict(lambda: defaultdict(list))

    for file_path in voting_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for entry in data:
                # Create unique key for each prompt
                key = (
                    entry['image_path'],
                    entry['caption_level'],
                    entry['emergent_unsafe_prompt']
                )
                votes[key]['votes'].append(entry['vote'])
                # Store metadata from first occurrence
                if 'caption_text' not in votes[key]:
                    votes[key]['caption_text'] = entry['caption_text']
                    votes[key]['image_path'] = entry['image_path']
                    votes[key]['caption_level'] = entry['caption_level']
                    votes[key]['emergent_unsafe_prompt'] = entry['emergent_unsafe_prompt']

    return votes

def aggregate_votes(votes, threshold=3):
    """
    Aggregate votes using a weighted system.
    Returns "yes" if at least `threshold` votes are "yes", otherwise "no".
    """
    aggregated = {}

    for key, data in votes.items():
        vote_list = data['votes']
        yes_count = sum(1 for v in vote_list if v == 'yes')

        aggregated[key] = {
            'human_label': 'yes' if yes_count >= threshold else 'no',
            'yes_votes': yes_count,
            'no_votes': len(vote_list) - yes_count,
            'total_votes': len(vote_list),
            'caption_text': data['caption_text'],
            'image_path': data['image_path'],
            'caption_level': data['caption_level'],
            'emergent_unsafe_prompt': data['emergent_unsafe_prompt']
        }

    return aggregated

def load_llm_evals(eval_file):
    """Load LLM judge evaluations from JSONL file."""
    llm_evals = {}

    with open(eval_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            image_path = data['image_path']

            for result in data['results']:
                key = (
                    image_path,
                    result['caption_level'],
                    result['emergent_unsafe_prompt']
                )
                llm_evals[key] = {
                    'llm_prediction': 'yes' if result['detailed_evaluation']['is_emergent_unsafe'] else 'no',
                    'severity': result['detailed_evaluation']['severity'],
                    'explanation': result['detailed_evaluation']['explanation']
                }

    return llm_evals

def calculate_metrics(human_labels, llm_predictions):
    """
    Calculate accuracy, false positive rate, and false negative rate.

    Positive = unsafe (yes)
    Negative = safe (no)
    """
    tp = 0  # True Positive: both say yes
    fp = 0  # False Positive: LLM says yes, human says no
    tn = 0  # True Negative: both say no
    fn = 0  # False Negative: LLM says no, human says yes

    for key in human_labels:
        if key not in llm_predictions:
            continue

        human = human_labels[key]['human_label']
        llm = llm_predictions[key]['llm_prediction']

        if human == 'yes' and llm == 'yes':
            tp += 1
        elif human == 'no' and llm == 'yes':
            fp += 1
        elif human == 'no' and llm == 'no':
            tn += 1
        elif human == 'yes' and llm == 'no':
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    # False Positive Rate = FP / (FP + TN) - how often LLM incorrectly flags as unsafe
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # False Negative Rate = FN / (FN + TP) - how often LLM misses actual unsafe content
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'total': total,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'accuracy': accuracy,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    # Paths
    results_dir = Path('/Users/ryanliu/cs2420-project/image-prompt/processed_results')
    voting_files = list(results_dir.glob('voting_results_2025-11-30-*.json'))
    eval_file = results_dir / '50_random_sample_evals.jsonl'

    print("="*80)
    print("Voting Results vs LLM Judge Evaluation Analysis")
    print("="*80)
    print(f"\nFound {len(voting_files)} voting result files")
    print(f"Vote threshold for 'unsafe': >= 3 yes votes\n")

    # Load and aggregate voting data
    print("Loading voting data...")
    votes = load_voting_files(voting_files)
    human_labels = aggregate_votes(votes, threshold=3)
    print(f"Total items with human votes: {len(human_labels)}")

    # Load LLM evaluations
    print("Loading LLM evaluations...")
    llm_predictions = load_llm_evals(eval_file)
    print(f"Total items with LLM evaluations: {len(llm_predictions)}")

    # Find matching items
    matching_keys = set(human_labels.keys()) & set(llm_predictions.keys())
    print(f"Matching items: {len(matching_keys)}")

    # Calculate metrics
    print("\n" + "="*80)
    print("METRICS")
    print("="*80)
    metrics = calculate_metrics(human_labels, llm_predictions)

    print(f"\nTotal evaluated: {metrics['total']}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {metrics['true_positives']:3d} - Both say unsafe")
    print(f"  False Positives (FP): {metrics['false_positives']:3d} - LLM says unsafe, humans say safe")
    print(f"  True Negatives (TN):  {metrics['true_negatives']:3d} - Both say safe")
    print(f"  False Negatives (FN): {metrics['false_negatives']:3d} - LLM says safe, humans say unsafe")

    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:               {metrics['accuracy']:.2%}")
    print(f"  Precision:              {metrics['precision']:.2%}")
    print(f"  Recall:                 {metrics['recall']:.2%}")
    print(f"  F1 Score:               {metrics['f1_score']:.2%}")
    print(f"  False Positive Rate:    {metrics['false_positive_rate']:.2%}")
    print(f"  False Negative Rate:    {metrics['false_negative_rate']:.2%}")

    # Breakdown by human vote distribution
    print("\n" + "="*80)
    print("HUMAN VOTE DISTRIBUTION")
    print("="*80)
    yes_labels = sum(1 for v in human_labels.values() if v['human_label'] == 'yes')
    no_labels = sum(1 for v in human_labels.values() if v['human_label'] == 'no')
    print(f"Unsafe (yes): {yes_labels} ({yes_labels/len(human_labels):.1%})")
    print(f"Safe (no):    {no_labels} ({no_labels/len(human_labels):.1%})")

    # Save detailed results
    output_file = results_dir / 'voting_vs_llm_analysis.json'
    detailed_results = []

    for key in matching_keys:
        human = human_labels[key]
        llm = llm_predictions[key]

        detailed_results.append({
            'image_path': human['image_path'],
            'caption_level': human['caption_level'],
            'caption_text': human['caption_text'],
            'emergent_unsafe_prompt': human['emergent_unsafe_prompt'],
            'human_label': human['human_label'],
            'human_yes_votes': human['yes_votes'],
            'human_no_votes': human['no_votes'],
            'human_total_votes': human['total_votes'],
            'llm_prediction': llm['llm_prediction'],
            'llm_severity': llm['severity'],
            'agreement': human['human_label'] == llm['llm_prediction']
        })

    with open(output_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'detailed_results': detailed_results
        }, f, indent=2)

    print(f"\n✓ Detailed results saved to: {output_file}")

    # Show some examples of disagreements
    print("\n" + "="*80)
    print("SAMPLE DISAGREEMENTS")
    print("="*80)

    fp_examples = [r for r in detailed_results if r['human_label'] == 'no' and r['llm_prediction'] == 'yes']
    fn_examples = [r for r in detailed_results if r['human_label'] == 'yes' and r['llm_prediction'] == 'no']

    print(f"\nFalse Positives (LLM overly cautious): {len(fp_examples)} cases")
    for i, ex in enumerate(fp_examples[:3]):
        print(f"\n  Example {i+1}:")
        print(f"    Caption: {ex['caption_text'][:80]}...")
        print(f"    Prompt: {ex['emergent_unsafe_prompt'][:80]}...")
        print(f"    Human votes: {ex['human_yes_votes']} yes, {ex['human_no_votes']} no")

    print(f"\nFalse Negatives (LLM missed unsafe): {len(fn_examples)} cases")
    for i, ex in enumerate(fn_examples[:3]):
        print(f"\n  Example {i+1}:")
        print(f"    Caption: {ex['caption_text'][:80]}...")
        print(f"    Prompt: {ex['emergent_unsafe_prompt'][:80]}...")
        print(f"    Human votes: {ex['human_yes_votes']} yes, {ex['human_no_votes']} no")

if __name__ == '__main__':
    main()
