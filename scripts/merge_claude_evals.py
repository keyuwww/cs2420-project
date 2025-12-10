#!/usr/bin/env python3
"""
Merge evaluation objects from claude_eval_new_evaluations.jsonl into
claude_eval_enhanced.jsonl by matching (image_path, prompt).

Writes output to final_evals/claude_eval_enhanced_merged.jsonl and
prints a short verification summary.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_PATH = ROOT / 'final_evals' / 'claude_eval_new_evaluations.jsonl'
ENH_PATH = ROOT / 'final_evals' / 'claude_eval_enhanced_merged.jsonl'
OUT_PATH = ROOT / 'final_evals' / 'claude_eval_enhanced_merged_merged.jsonl'


def load_mapping(new_path):
    mapping = {}
    with new_path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img = obj.get('image_path')
            prompt = obj.get('emergent_prompt')
            # normalize key to tuple
            key = (img, prompt)
            mapping[key] = obj.get('evaluation')
    return mapping


def merge(enh_path, mapping, out_path):
    total = 0
    replaced = 0
    null_remaining = 0
    missing_keys = []
    with enh_path.open('r', encoding='utf-8') as inf, out_path.open('w', encoding='utf-8') as outf:
        for line in inf:
            total += 1
            line = line.rstrip('\n')
            if not line:
                continue
            obj = json.loads(line)
            # detect which prompt field exists in this enhanced file
            prompt = None
            if 'emergent_prompt' in obj:
                prompt = obj.get('emergent_prompt')
            elif 'emergent_unsafe_prompt' in obj:
                prompt = obj.get('emergent_unsafe_prompt')
            key = (obj.get('image_path'), prompt)

            if obj.get('evaluation') is None:
                ev = mapping.get(key)
                if ev is not None:
                    obj['evaluation'] = ev
                    replaced += 1
                else:
                    null_remaining += 1
                    missing_keys.append(key)

            # write compact JSONL line
            outf.write(json.dumps(obj, ensure_ascii=False) + '\n')

    return {
        'total_lines': total,
        'replaced': replaced,
        'null_remaining': null_remaining,
        'missing_examples': missing_keys[:20],
        'out_path': str(out_path)
    }


def verify(out_path):
    nulls = 0
    lines = 0
    with out_path.open('r', encoding='utf-8') as f:
        for line in f:
            lines += 1
            obj = json.loads(line)
            if obj.get('evaluation') is None:
                nulls += 1
    return lines, nulls


def main():
    mapping = load_mapping(NEW_PATH)
    print(f'Loaded mapping entries: {len(mapping)}')
    res = merge(ENH_PATH, mapping, OUT_PATH)
    print('Merge result:')
    print(json.dumps({k: v for k, v in res.items() if k != 'missing_examples'}, indent=2))
    if res['missing_examples']:
        print('Sample missing keys:', res['missing_examples'])
    lines, nulls = verify(OUT_PATH)
    print(f'Output file: {OUT_PATH}  lines={lines}  null_evaluations={nulls}')
    # exit non-zero if there are null remaining or line count mismatches
    if nulls != 0 or lines != 3988:
        print('\nVerification FAILED: either null evaluations remain or line count != 3988')
        raise SystemExit(2)
    print('\nVerification PASSED: no null evaluations and line count == 3988')


if __name__ == '__main__':
    main()
