#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import logging
import argparse
from typing import Dict, List
import sys

from transformers import AutoProcessor, AutoModel
from peft import get_peft_model, LoraConfig, PeftModel
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmergentUnsafetyHead(nn.Module):
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
        h = h.float()
        return self.mlp(h)


def load_data(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class EmergentUnsafetyDataset(Dataset):
    def __init__(self, data: List[Dict], image_dir: str, processor):
        self.data = data
        self.image_dir = Path(image_dir)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, processor):
    images = []
    prompts = []
    labels = []
    confidences = []
    
    for item in batch:
        image_path_str = item.get("image_path", "")
        if Path(image_path_str).is_absolute():
            image_path = Path(image_path_str)
        else:
            image_path = Path(item.get("image_path", ""))
        
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            original_prompt = item.get('prompt', '')
            prompts.append(f"USER: <image>\n{original_prompt}\nASSISTANT:")
            labels.append(int(item.get("label", 0)))
            confidences.append(float(item.get("confidence", 1.0)))
    
    if len(images) == 0:
        return None
    
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    
    inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
    inputs["confidence"] = torch.tensor(confidences, dtype=torch.float32)
    
    return inputs


def extract_multimodal_embedding(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        embedding = last_hidden[:, -1, :]
    
    return embedding


def evaluate(model, safety_head, data_loader, device, threshold=0.5):
    model.eval()
    safety_head.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if batch is None:
                continue
            
            embedding = extract_multimodal_embedding(model, batch, device)
            p_emergent = safety_head(embedding)
            
            all_preds.extend(p_emergent.cpu().numpy().flatten().tolist())
            all_labels.extend(batch["labels"].cpu().numpy().flatten().tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    binary_preds = (all_preds >= threshold).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    auroc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0
    
    safe_mask = all_labels == 0
    unsafe_mask = all_labels == 1
    safe_acc = accuracy_score(all_labels[safe_mask], binary_preds[safe_mask]) if safe_mask.sum() > 0 else 0.0
    unsafe_acc = accuracy_score(all_labels[unsafe_mask], binary_preds[unsafe_mask]) if unsafe_mask.sum() > 0 else 0.0
    
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "safe_accuracy": safe_acc,
        "unsafe_accuracy": unsafe_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


def find_best_threshold(val_preds, val_labels, metric='f1'):
    val_preds = np.array(val_preds) if not isinstance(val_preds, np.ndarray) else val_preds
    val_labels = np.array(val_labels) if not isinstance(val_labels, np.ndarray) else val_labels
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        binary_preds = (val_preds >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(val_labels, binary_preds, zero_division=0)
        elif metric == 'accuracy':
            score = (binary_preds == val_labels).mean()
        elif metric == 'balanced_accuracy':
            safe_mask = val_labels == 0
            unsafe_mask = val_labels == 1
            
            if isinstance(safe_mask, np.ndarray) and isinstance(unsafe_mask, np.ndarray):
                if safe_mask.sum() > 0 and unsafe_mask.sum() > 0:
                    specificity = (binary_preds[safe_mask] == 0).mean()
                    sensitivity = (binary_preds[unsafe_mask] == 1).mean()
                    score = (specificity + sensitivity) / 2
                else:
                    score = 0
            else:
                score = 0
        else:
            score = f1_score(val_labels, binary_preds, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def main():
    parser = argparse.ArgumentParser(description="Complete post-training threshold tuning and evaluation")
    parser.add_argument("--checkpoint-dir", type=str,
                       default="/n/netscratch/dam_lab/Lab/sliaw/cs2432_artifacts/lora_checkpoints",
                       help="Directory containing checkpoints")
    parser.add_argument("--model-name", type=str, default="llava-hf/llava-1.5-7b-hf",
                       help="Base model name")
    parser.add_argument("--cache-dir", type=str,
                       default="/n/netscratch/dam_lab/Lab/sliaw/cs2432_artifacts",
                       help="Cache directory")
    parser.add_argument("--image-dir", type=str, default="/n/home10/sliaw/cs2420-project",
                       help="Image directory")
    parser.add_argument("--val-file", type=str,
                       default="/n/home10/sliaw/cs2420-project/llava/lora_data/bench_val.fixed.jsonl",
                       help="Validation data file")
    parser.add_argument("--test-file", type=str,
                       default="/n/home10/sliaw/cs2420-project/llava/lora_data/bench_test.fixed.jsonl",
                       help="Test data file")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="HuggingFace token")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        logger.info("✓ Authenticated with Hugging Face")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    logger.info(f"Loading processor for {args.model_name}...")
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
    )
    logger.info("✓ Processor loaded")
    
    logger.info(f"Loading base model {args.model_name}...")
    base_model = AutoModel.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("✓ Base model loaded")
    
    lora_adapter_path = checkpoint_dir / "lora_adapter"
    if lora_adapter_path.exists():
        logger.info(f"Loading LoRA adapter from {lora_adapter_path}...")
        model = PeftModel.from_pretrained(base_model, str(lora_adapter_path))
        logger.info("✓ LoRA adapter loaded")
    else:
        checkpoints = list(checkpoint_dir.glob("best_checkpoint_epoch*.pt"))
        if not checkpoints:
            logger.error("No checkpoints found!")
            return
        
        checkpoints.sort(key=lambda x: int(x.stem.split("epoch")[-1]))
        best_checkpoint_path = checkpoints[-1]
        
        logger.info(f"Loading checkpoint from {best_checkpoint_path}...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(base_model, peft_config)
        model.load_state_dict(checkpoint["model_state"])
        logger.info("✓ LoRA model loaded from checkpoint")
    
    model.eval()
    
    safety_head_path = checkpoint_dir / "safety_head.pt"
    if not safety_head_path.exists():
        logger.error(f"Safety head not found at {safety_head_path}")
        return
    
    logger.info(f"Loading safety head from {safety_head_path}...")
    safety_head = EmergentUnsafetyHead(hidden_dim=4096, dropout=0.1)
    safety_head.load_state_dict(torch.load(safety_head_path, map_location=device, weights_only=False))
    safety_head = safety_head.to(device)
    safety_head.eval()
    logger.info("✓ Safety head loaded")
    
    logger.info(f"Loading validation data from {args.val_file}...")
    val_data = load_data(args.val_file)
    logger.info(f"Loaded {len(val_data)} validation samples")
    
    logger.info(f"Loading test data from {args.test_file}...")
    test_data = load_data(args.test_file)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    from functools import partial
    collate_fn_with_processor = partial(collate_fn, processor=processor)
    
    val_loader = DataLoader(
        EmergentUnsafetyDataset(val_data, args.image_dir, processor),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_processor,
        num_workers=0
    )
    
    test_loader = DataLoader(
        EmergentUnsafetyDataset(test_data, args.image_dir, processor),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_processor,
        num_workers=0
    )
    
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD TUNING ON VALIDATION SET")
    logger.info("="*60)
    
    val_metrics = evaluate(model, safety_head, val_loader, device, threshold=0.5)
    val_preds = val_metrics['all_preds']
    val_labels = val_metrics['all_labels']
    
    logger.info(f"Validation predictions shape: {val_preds.shape}")
    logger.info(f"Validation labels shape: {val_labels.shape}")
    
    best_threshold_f1, best_f1 = find_best_threshold(val_preds, val_labels, metric='f1')
    best_threshold_balanced, best_balanced = find_best_threshold(val_preds, val_labels, metric='balanced_accuracy')
    
    logger.info(f"\nThreshold tuning results:")
    logger.info(f"  Best F1 threshold: {best_threshold_f1:.3f} (F1={best_f1:.4f})")
    logger.info(f"  Best balanced accuracy threshold: {best_threshold_balanced:.3f} (Bal.Acc={best_balanced:.4f})")
    
    best_threshold = best_threshold_balanced
    logger.info(f"\nUsing threshold: {best_threshold:.3f}")
    
    threshold_path = checkpoint_dir / "best_threshold.txt"
    with open(threshold_path, 'w') as f:
        f.write(str(best_threshold))
    logger.info(f"✓ Saved threshold to {threshold_path}")
    
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("="*60)
    
    test_metrics = evaluate(model, safety_head, test_loader, device, threshold=best_threshold)
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  AUROC: {test_metrics['auroc']:.4f}")
    logger.info(f"  Safe Accuracy: {test_metrics['safe_accuracy']:.4f}")
    logger.info(f"  Unsafe Accuracy: {test_metrics['unsafe_accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"  Threshold: {test_metrics['threshold']:.3f}")
    
    final_results = {
        "threshold_tuning": {
            "best_threshold_f1": float(best_threshold_f1),
            "best_f1": float(best_f1),
            "best_threshold_balanced": float(best_threshold_balanced),
            "best_balanced_acc": float(best_balanced),
            "selected_threshold": float(best_threshold),
        },
        "test": {
            "accuracy": float(test_metrics['accuracy']),
            "auroc": float(test_metrics['auroc']),
            "safe_accuracy": float(test_metrics['safe_accuracy']),
            "unsafe_accuracy": float(test_metrics['unsafe_accuracy']),
            "precision": float(test_metrics['precision']),
            "recall": float(test_metrics['recall']),
            "f1": float(test_metrics['f1']),
            "threshold": float(test_metrics['threshold']),
        }
    }
    
    results_path = checkpoint_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"\n✓ Saved final results to {results_path}")
    
    logger.info("\n" + "="*60)
    logger.info("POST-TRAINING COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

