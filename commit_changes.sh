#!/bin/bash
# Script to commit and push changes (excluding sbatch files)

cd /n/home10/sliaw/cs2420-project

# Stage all Python files in llava directory
git add llava/*.py

# Stage baseline_results (excluding .md files that aren't README.md)
git add llava/baseline_results/*.json llava/baseline_results/*.txt llava/baseline_results/*.jsonl llava/baseline_results/plots/

# Make sure sbatch files are NOT staged
git reset HEAD llava/*.sbatch 2>/dev/null

# Check what's staged
echo "Files staged for commit:"
git status --short | grep "^A\|^M"

# Commit
git commit -m "Update scripts to use split files instead of split ratios, remove non-README .md files"

# Push
git push origin task/SL-001



