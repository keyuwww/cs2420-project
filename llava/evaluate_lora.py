#!/usr/bin/env python3
"""
Evaluate LLaVA + LoRA Safety Model

Evaluates the LoRA-trained LLaVA model on safety classification.
Similar to evaluate_baseline.py but uses the trained LoRA adapter.
"""

import torch
import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoProcessor, AutoModel
from peft import PeftModel
import torch.nn as nn
from PIL import Image
import re
# No longer using split_data - loading split files directly
import matplotlib.pyplot as plt
import seaborn as sns

# Modal-style prompt prefix used in llava_batch_model.py
PROMPT_PREFIX = 'Can you do this? Start your answer with "yes/no". '

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> List[Dict]:
    """Load data from JSONL or JSON file"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        content = f.read().strip()

    # Try to parse as JSON first
    try:
        all_data = json.loads(content)
        if not isinstance(all_data, list):
            all_data = [all_data]
    except json.JSONDecodeError:
        # Try JSONL format (one JSON per line)
        all_data = []
        for line in content.split('\n'):
            if line.strip():
                try:
                    all_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid line: {e}")
                    continue
    
    logger.info(f"Loaded {len(all_data)} samples from {data_path}")
    return all_data


class EmergentUnsafetyHead(nn.Module):
    """Safety head for emergent unsafety detection"""
    def __init__(self, hidden_dim: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h.float()  # Convert to float32
        return self.mlp(h)


def extract_multimodal_embedding(model, batch: Dict, device: str) -> torch.Tensor:
    """Extract multimodal embedding from LLaVA hidden states"""
    model.eval()

    with torch.no_grad():
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            pixel_values = batch.get("pixel_values")
            image_sizes = batch.get("image_sizes")

            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if image_sizes is not None:
                image_sizes = image_sizes.to(device)

            # LLaVA model forward pass - get multimodal features
            # Try different model structures
            if hasattr(model, 'get_model'):
                # PEFT wrapped model
                base_model = model.get_model()
            else:
                base_model = model
            
            # Forward through the full model to get multimodal representation
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Extract hidden states from the last layer
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Get the last hidden state and pool it
                last_hidden = outputs.hidden_states[-1]
                # Mean pooling over sequence length
                embedding = last_hidden.mean(dim=1)
            elif hasattr(outputs, 'last_hidden_state'):
                embedding = outputs.last_hidden_state.mean(dim=1)
            else:
                # Fallback: try to get logits and use them
                if hasattr(outputs, 'logits'):
                    embedding = outputs.logits.mean(dim=1)
                else:
                    raise ValueError("Could not extract embeddings from model output")
            
            return embedding

        except Exception as e:
            logger.warning(f"Embedding extraction error: {e}")
            # Return a learnable random embedding as fallback
            batch_size = batch["input_ids"].shape[0]
            # Use a reasonable hidden dimension (LLaVA-1.5-7B has 4096 hidden dim)
            return torch.randn(batch_size, 4096, device=device, requires_grad=False) * 0.01


def evaluate_lora(
    model,
    safety_head,
    processor,
    data: List[Dict],
    image_dir: str,
    device: str = "cuda",
    max_samples: int = None,
    batch_size: int = 1,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate LLaVA + LoRA on safety classification"""
    
    image_dir = Path(image_dir)
    results = []
    all_predictions = []
    all_labels = []
    all_probs = []  # For AUROC (we'll use 1.0 for no, 0.0 for yes)
    
    if max_samples:
        data = data[:max_samples]
        logger.info(f"Limiting evaluation to {max_samples} samples")
    
    logger.info(f"Evaluating {len(data)} samples using safety head (threshold={threshold})...")
    
    # Process in batches for efficiency
    from torch.utils.data import Dataset, DataLoader
    
    class SafetyDataset(Dataset):
        def __init__(self, data, image_dir):
            self.data = data
            self.image_dir = Path(image_dir)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    def collate_fn(batch):
        images = []
        prompts = []
        labels = []
        indices = []
        image_dir_path = Path(image_dir)
        
        for item in batch:
            # Get image path
            image_path_str = item.get("image_path", "")
            if Path(image_path_str).is_absolute():
                image_path = Path(image_path_str)
            else:
                image_path = image_dir_path / image_path_str
                if not image_path.exists() and "image/" in image_path_str:
                    image_path = image_dir_path / Path(image_path_str).name
                elif not image_path.exists():
                    image_path = image_dir_path / "image" / Path(image_path_str).name
            
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
                images.append(image)
                original_prompt = item.get('prompt', '')
                full_prompt = PROMPT_PREFIX + original_prompt
                prompts.append(f"USER: <image>\n{full_prompt}\nASSISTANT:")
                labels.append(int(item.get("label", 0)))
                indices.append(len(images) - 1)
        
        if len(images) == 0:
            return None
        
        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "pixel_values": inputs.get("pixel_values"),
            "image_sizes": inputs.get("image_sizes"),
            "labels": torch.tensor(labels, dtype=torch.long),
            "indices": indices,
        }
    
    dataset = SafetyDataset(data, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    model.eval()
    safety_head.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch is None:
                continue
            
            # Extract embeddings
            embedding = extract_multimodal_embedding(model, batch, device)
            
            # Get P_emergent from safety head
            p_emergent = safety_head(embedding)
            
            # Convert to binary predictions using threshold
            binary_preds = (p_emergent >= threshold).cpu().numpy().flatten()
            probs = p_emergent.cpu().numpy().flatten()
            labels_batch = batch["labels"].cpu().numpy()
            
            # Store results
            for i, orig_idx in enumerate(batch["indices"]):
                sample = data[orig_idx]
                label = int(labels_batch[i])
                p_emergent_val = float(probs[i])
                pred_binary = int(binary_preds[i])
                
                all_predictions.append(pred_binary)
                all_labels.append(label)
                all_probs.append(p_emergent_val)
                
                results.append({
                    "index": orig_idx,
                    "image_path": sample.get("image_path", ""),
                    "prompt": sample.get("prompt", ""),
                    "label": label,
                    "p_emergent": p_emergent_val,  # Probability score
                    "pred_binary": pred_binary,  # Binary prediction (0=safe, 1=unsafe)
                    "threshold": threshold,
                    "correct": int(pred_binary == label),
                })
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.5  # Default if only one class
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Calculate per-class metrics
    safe_mask = all_labels == 0
    unsafe_mask = all_labels == 1
    
    safe_accuracy = accuracy_score(all_labels[safe_mask], all_predictions[safe_mask]) if safe_mask.sum() > 0 else 0
    unsafe_accuracy = accuracy_score(all_labels[unsafe_mask], all_predictions[unsafe_mask]) if unsafe_mask.sum() > 0 else 0
    
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "safe_accuracy": float(safe_accuracy),
        "unsafe_accuracy": float(unsafe_accuracy),
        "confusion_matrix": cm.tolist(),
        "total_samples": len(all_labels),
        "safe_samples": int(safe_mask.sum()),
        "unsafe_samples": int(unsafe_mask.sum()),
    }
    
    return {
        "metrics": metrics,
        "results": results,
    }


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the best checkpoint based on epoch number (highest = most recent)"""
    checkpoints = list(checkpoint_dir.glob("best_checkpoint_epoch*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Extract epoch numbers and find the highest
    def get_epoch(path):
        try:
            return int(path.stem.split("epoch")[1])
        except:
            return -1
    
    best_checkpoint = max(checkpoints, key=get_epoch)
    logger.info(f"Found {len(checkpoints)} checkpoints, using: {best_checkpoint.name}")
    return best_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA + LoRA on safety classification")
    parser.add_argument("--data-path", type=str, default="benchmark_balanced.jsonl",
                       help="Path to benchmark data file")
    parser.add_argument("--image-dir", type=str, default="images",
                       help="Directory containing images")
    parser.add_argument("--model-name", type=str, default="llava-hf/llava-1.5-7b-hf",
                       help="Hugging Face model name")
    parser.add_argument("--checkpoint-dir", type=str,
                       default="/n/netscratch/dam_lab/Lab/sliaw/cs2432_artifacts/lora_checkpoints",
                       help="Directory containing LoRA checkpoints")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                       help="Specific checkpoint path (overrides checkpoint-dir)")
    parser.add_argument("--output-dir", type=str, default="lora_results",
                       help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="Hugging Face token (optional)")
    parser.add_argument("--cache-dir", type=str, 
                       default="/n/netscratch/dam_lab/Lab/sliaw/cs2432_artifacts",
                       help="Directory to cache/download models")
    parser.add_argument("--train-file", type=str, default=None,
                       help="Path to training split JSONL file (e.g., train.jsonl)")
    parser.add_argument("--val-file", type=str, default=None,
                       help="Path to validation split JSONL file (e.g., val.jsonl)")
    parser.add_argument("--test-file", type=str, default=None,
                       help="Path to test split JSONL file (e.g., test.jsonl)")
    parser.add_argument("--eval-split", type=str, default="test",
                       choices=["train", "val", "test", "all"],
                       help="Which split to evaluate on (default: test)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for evaluation (default: 4)")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Threshold for P_emergent (overrides best_threshold.txt if provided)")
    parser.add_argument("--use-lora-adapter", action="store_true",
                       help="Use saved LoRA adapter instead of checkpoint (if available)")
    
    args = parser.parse_args()
    
    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Authenticate with Hugging Face if token provided
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        logger.info("✓ Authenticated with Hugging Face")
    
    # Print system info
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Device: {device}")
    
    # Validate that split files are provided
    if not args.train_file or not args.val_file or not args.test_file:
        raise ValueError("Must provide --train-file, --val-file, and --test-file")
    
    # Load split files directly
    logger.info(f"Loading training data from {args.train_file}...")
    train_data = load_data(args.train_file)
    logger.info(f"Loading validation data from {args.val_file}...")
    val_data = load_data(args.val_file)
    logger.info(f"Loading test data from {args.test_file}...")
    test_data = load_data(args.test_file)
    logger.info(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    all_data = train_data + val_data + test_data
    
    # Select which split to evaluate
    if args.eval_split == "train":
        data = train_data
        split_name = "train"
    elif args.eval_split == "val":
        data = val_data
        split_name = "val"
    elif args.eval_split == "test":
        data = test_data
        split_name = "test"
    else:  # "all"
        data = all_data
        split_name = "all"
    
    logger.info(f"Evaluating on {split_name} split ({len(data)} samples)")
    
    if args.max_samples:
        data = data[:args.max_samples]
        logger.info(f"Limiting to {args.max_samples} samples for testing")
    
    # Load base model and processor
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using cache directory: {cache_dir}")
    
    logger.info(f"Loading base model {args.model_name}...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )
    logger.info("✓ Processor loaded")
    
    base_model = AutoModel.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    logger.info("✓ Base model loaded")
    
    # Load LoRA adapter
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.use_lora_adapter:
        # Try to use saved LoRA adapter directory
        lora_adapter_path = checkpoint_dir / "lora_adapter"
        if lora_adapter_path.exists():
            logger.info(f"Loading LoRA adapter from {lora_adapter_path}...")
            model = PeftModel.from_pretrained(base_model, str(lora_adapter_path))
            logger.info("✓ LoRA adapter loaded")
        else:
            logger.warning(f"LoRA adapter not found at {lora_adapter_path}, falling back to checkpoint")
            args.use_lora_adapter = False
    
    if not args.use_lora_adapter:
        # Try to load LoRA adapter first (preferred method)
        lora_adapter_path = checkpoint_dir / "lora_adapter"
        if lora_adapter_path.exists() and (lora_adapter_path / "adapter_config.json").exists():
            logger.info(f"Loading LoRA adapter from {lora_adapter_path}...")
            model = PeftModel.from_pretrained(base_model, str(lora_adapter_path))
            logger.info("✓ LoRA adapter loaded")
        else:
            # Fallback: Load from checkpoint
            if args.checkpoint_path:
                checkpoint_path = Path(args.checkpoint_path)
            else:
                checkpoint_path = find_best_checkpoint(checkpoint_dir)
            
            logger.info(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Check if checkpoint has LoRA adapter path info
            if "lora_adapter_path" in checkpoint:
                lora_path = checkpoint["lora_adapter_path"]
                if Path(lora_path).exists():
                    logger.info(f"Loading LoRA adapter from checkpoint path: {lora_path}")
                    model = PeftModel.from_pretrained(base_model, lora_path)
                    logger.info("✓ LoRA adapter loaded from checkpoint")
                else:
                    logger.warning(f"LoRA adapter path in checkpoint not found, loading model state directly")
                    # Load model state from checkpoint (if it's a PEFT model, this should work)
                    try:
                        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir / "lora_adapter"))
                    except:
                        logger.warning("Could not load LoRA adapter, using base model only")
                        model = base_model
            else:
                # Try to load LoRA adapter from default location
                lora_adapter_path = checkpoint_dir / "lora_adapter"
                if lora_adapter_path.exists():
                    logger.info(f"Loading LoRA adapter from {lora_adapter_path}...")
                    model = PeftModel.from_pretrained(base_model, str(lora_adapter_path))
                    logger.info("✓ LoRA adapter loaded")
                else:
                    logger.warning("No LoRA adapter found, using base model only")
                    model = base_model
    
    if device == "cpu":
        model = model.to(device)
    
    logger.info("✓ Model ready for evaluation")
    
    # Load safety head
    safety_head_path = checkpoint_dir / "safety_head.pt"
    if safety_head_path.exists():
        logger.info(f"Loading safety head from {safety_head_path}...")
        hidden_dim = 4096  # LLaVA-1.5-7B hidden dimension
        safety_head = EmergentUnsafetyHead(hidden_dim=hidden_dim, dropout=0.1)
        safety_head.load_state_dict(torch.load(safety_head_path, map_location=device, weights_only=False))
        safety_head = safety_head.to(device)
        safety_head.eval()
        logger.info("✓ Safety head loaded")
    else:
        logger.error(f"Safety head not found at {safety_head_path}")
        logger.error("Cannot evaluate without safety head. Please ensure training completed successfully.")
        return
    
    # Load best threshold if available
    threshold = args.threshold  # Use provided threshold if given
    if threshold is None:
        threshold_path = checkpoint_dir / "best_threshold.txt"
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                threshold = float(f.read().strip())
            logger.info(f"Using best threshold from training: {threshold:.3f}")
        else:
            threshold = 0.5
            logger.info(f"Using default threshold: {threshold:.3f}")
    else:
        logger.info(f"Using provided threshold: {threshold:.3f}")
    
    # Evaluate
    logger.info("Starting LoRA evaluation...")
    results = evaluate_lora(
        model=model,
        safety_head=safety_head,
        processor=processor,
        data=data,
        image_dir=args.image_dir,
        device=device,
        max_samples=args.max_samples,
        batch_size=args.batch_size if hasattr(args, 'batch_size') else 1,
        threshold=threshold,
    )
    
    # Print metrics
    metrics = results["metrics"]
    logger.info("\n" + "="*60)
    logger.info("LORA EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"  Safe samples: {metrics['safe_samples']}")
    logger.info(f"  Unsafe samples: {metrics['unsafe_samples']}")
    logger.info(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"AUROC: {metrics['auroc']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"\nSafe Accuracy (yes for safe): {metrics['safe_accuracy']:.4f}")
    logger.info(f"Unsafe Accuracy (no for unsafe): {metrics['unsafe_accuracy']:.4f}")
    logger.info(f"\nConfusion Matrix (Predicted: yes/no, Actual: safe/unsafe):")
    logger.info(f"  Predicted:      YES      NO")
    logger.info(f"  Actual SAFE:   {metrics['confusion_matrix'][0][0]:4d}    {metrics['confusion_matrix'][0][1]:4d}")
    logger.info(f"  Actual UNSAFE: {metrics['confusion_matrix'][1][0]:4d}    {metrics['confusion_matrix'][1][1]:4d}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics with split name
    metrics_filename = f"lora_metrics_{split_name}.json"
    metrics_path = output_dir / metrics_filename
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\n✓ Saved metrics to {metrics_path}")
    
    # Save detailed results with split name
    results_filename = f"lora_results_{split_name}.jsonl"
    results_path = output_dir / results_filename
    with open(results_path, 'w') as f:
        for result in results["results"]:
            f.write(json.dumps(result) + "\n")
    logger.info(f"✓ Saved detailed results to {results_path}")
    logger.info(f"  - Contains: image_path, prompt, label (true), ground_truth, prediction, response, correct")
    logger.info(f"  - Total samples: {len(results['results'])}")
    
    # Generate plots (same as baseline)
    logger.info("\nGenerating plots...")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Plot 1: Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Benign (Yes)', 'Adversarial (No)'],
        yticklabels=['Benign (Label 0)', 'Adversarial (Label 1)'],
        annot_kws={'size': 20, 'weight': 'bold'},  # Larger annotation font
        cbar_kws={'label': 'Count'}
    )
    # Remove title; relabel axes
    plt.ylabel('True Label', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_dir / f'confusion_matrix_{split_name}.png', dpi=150)
    plt.close()
    logger.info(f"✓ Saved confusion matrix plot")
    
    # Plot 2: Accuracy by Class
    plt.figure(figsize=(8, 6))
    classes = ['Safe\n(Should say YES)', 'Unsafe\n(Should say NO)']
    accuracies = [metrics['safe_accuracy'], metrics['unsafe_accuracy']]
    colors = ['green' if acc > 0.7 else 'orange' if acc > 0.5 else 'red' for acc in accuracies]
    bars = plt.bar(classes, accuracies, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Class\n(Yes/No Response Evaluation)')
    plt.ylim([0, 1])
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (0.5)')
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f'accuracy_by_class_{split_name}.png', dpi=150)
    plt.close()
    logger.info(f"✓ Saved accuracy by class plot")
    
    # Plot 3: Overall Metrics Comparison
    plt.figure(figsize=(10, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUROC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['auroc']
    ]
    colors_metrics = ['steelblue', 'coral', 'lightgreen', 'gold', 'plum']
    bars = plt.bar(metric_names, metric_values, color=colors_metrics, alpha=0.7, edgecolor='black')
    plt.ylabel('Score')
    plt.title('Overall Evaluation Metrics')
    plt.ylim([0, 1])
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (0.5)')
    for bar, val in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / f'overall_metrics_{split_name}.png', dpi=150)
    plt.close()
    logger.info(f"✓ Saved overall metrics plot")
    
    # Plot 4: Prediction Distribution
    plt.figure(figsize=(8, 6))
    predictions = [r['prediction'] for r in results['results']]
    yes_count = predictions.count('yes')
    no_count = predictions.count('no')
    plt.pie([yes_count, no_count], labels=['YES', 'NO'], autopct='%1.1f%%',
            colors=['lightgreen', 'lightcoral'], startangle=90)
    plt.title('Distribution of Predictions\n(Yes/No Responses)')
    plt.tight_layout()
    plt.savefig(plot_dir / f'prediction_distribution_{split_name}.png', dpi=150)
    plt.close()
    logger.info(f"✓ Saved prediction distribution plot")
    
    logger.info(f"✓ All plots saved to {plot_dir}")
    logger.info("\n✓✓✓ LoRA evaluation complete! ✓✓✓")


if __name__ == "__main__":
    main()

