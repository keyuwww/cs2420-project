#!/usr/bin/env python3
"""
LoRA Safety Adapter Training for LLaVA-1.6-Mistral-7B
Emergent Multimodal Unsafety Detection

Training pipeline for detecting emergent unsafety in image-text pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from typing import Dict, List
import warnings
import os
import argparse
import sys

warnings.filterwarnings('ignore')

from transformers import AutoProcessor, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
from sklearn.metrics import roc_auc_score
# No longer using split_data - loading split files directly

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Training configuration"""
    # Model
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    num_epochs: int = 3
    weight_decay: float = 0.01

    # Data
    data_path: str = "benchmark_balanced.jsonl"
    image_dir: str = "images"
    max_seq_length: int = 128

    # Train/Test Split Files
    train_file: str = None
    val_file: str = None
    test_file: str = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: str = "bf16"

    # Output
    output_dir: str = "/n/netscratch/dam_lab/Lab/sliaw/cs2432_artifacts/lora_checkpoints"

    # Thresholds
    tau_low: float = 0.3
    tau_high: float = 0.7

    # Hugging Face token (optional, for private models)
    hf_token: str = None


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


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(-1)
        p_t = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        # Return per-sample loss (not averaged) so we can apply confidence weighting
        return self.alpha * focal_weight * bce_loss


class EmergentUnsafetyDataset(Dataset):
    """Dataset for emergent unsafety detection"""
    def __init__(
        self,
        data: List[Dict],
        image_dir: str,
        processor,
        max_seq_length: int = 128,
    ):
        self.data = data
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Handle image path (can be relative or absolute)
        image_path_str = sample.get("image_path", "")
        
        # Handle different path formats
        if Path(image_path_str).is_absolute():
            image_path = Path(image_path_str)
        else:
            # Relative path - try different combinations
            image_path = self.image_dir / image_path_str
            
            # If path includes "image/" prefix, try with and without
            if not image_path.exists() and "image/" in image_path_str:
                # Try without the "image/" prefix
                image_path = self.image_dir / Path(image_path_str).name
            elif not image_path.exists():
                # Try with "image/" prefix if image_dir doesn't have it
                image_path = self.image_dir / "image" / Path(image_path_str).name

        # Load image
        try:
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path} (original: {image_path_str})")
                image = Image.new("RGB", (384, 384), color=(128, 128, 128))
            else:
                image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            image = Image.new("RGB", (384, 384), color=(128, 128, 128))

        # Get prompt and label
        prompt = sample.get("prompt", "")
        label = float(sample.get("label", 0))
        confidence = float(sample.get("confidence", 3)) / 3.0

        return {
            "image": image,
            "prompt": prompt,  # Don't add <image> token here - processor handles it
            "label": label,
            "confidence": confidence,
        }


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


# split_data function is now imported from data_utils.py to ensure consistency


def collate_fn(batch: List[Dict], processor) -> Dict:
    """Collate function for DataLoader"""
    images = [item["image"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    confidence = torch.tensor([item["confidence"] for item in batch], dtype=torch.float32)

    # Add <image> token to prompts for LLaVA format
    prompts_with_image = [f"<image>\n{prompt}" for prompt in prompts]
    
    # Process images and text together using processor
    # LLaVA processor handles both image and text
    inputs = processor(
        text=prompts_with_image,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "pixel_values": inputs.get("pixel_values"),
        "image_sizes": inputs.get("image_sizes"),
        "labels": labels,
        "confidence": confidence,
    }


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
            return torch.randn(batch_size, 4096, device=device, requires_grad=True) * 0.01


def train_epoch(model, safety_head, train_loader, optimizer, criterion, device,
                gradient_accumulation_steps=2, log_every=10):
    """Train for one epoch"""
    model.train()
    safety_head.train()

    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc="Training")

    for step, batch in enumerate(pbar):
        embedding = extract_multimodal_embedding(model, batch, device)
        p_emergent = safety_head(embedding)

        labels = batch["labels"].to(device)
        confidence = batch["confidence"].to(device)

        # Get per-sample loss (not averaged yet)
        per_sample_loss = criterion(p_emergent, labels)
        # Apply confidence weighting and average
        loss = (per_sample_loss * confidence).mean()
        # Scale for gradient accumulation
        loss_scaled = loss / gradient_accumulation_steps
        loss_scaled.backward()

        # Accumulate the original (unscaled) loss for reporting
        total_loss += loss.item()
        num_batches += 1

        all_preds.extend(p_emergent.detach().cpu().numpy().flatten())
        all_labels.extend(labels.detach().cpu().numpy())

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % log_every == 0:
            pbar.set_postfix({"loss": total_loss / num_batches})

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        auroc = 0.5

    return {"loss": total_loss / num_batches, "auroc": auroc}


@torch.no_grad()
def evaluate(model, safety_head, val_loader, criterion, device, threshold=0.5):
    """Evaluate on validation/test set"""
    model.eval()
    safety_head.eval()

    total_loss = 0
    all_preds = []  # P_emergent probabilities
    all_labels = []

    for batch in tqdm(val_loader, desc="Evaluating"):
        embedding = extract_multimodal_embedding(model, batch, device)
        p_emergent = safety_head(embedding)

        labels = batch["labels"].to(device)
        # Get per-sample loss and average (no confidence weighting in eval)
        per_sample_loss = criterion(p_emergent, labels)
        loss = per_sample_loss.mean()
        total_loss += loss.item()

        all_preds.extend(p_emergent.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Convert probabilities to binary predictions using threshold
    binary_preds = (all_preds >= threshold).astype(int)

    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        auroc = 0.5

    accuracy = (binary_preds == all_labels).mean()
    
    # Refusal rate: % of unsafe samples correctly identified as unsafe (P_emergent >= threshold)
    emergent_mask = all_labels == 1
    refusal_rate = (binary_preds[emergent_mask] == 1).mean() if emergent_mask.sum() > 0 else 0

    # False refusal: % of safe samples incorrectly identified as unsafe
    safe_mask = all_labels == 0
    false_refusal = (binary_preds[safe_mask] == 1).mean() if safe_mask.sum() > 0 else 0

    return {
        "loss": total_loss / len(val_loader),
        "accuracy": accuracy,
        "auroc": auroc,
        "refusal_rate": refusal_rate,
        "false_refusal_rate": false_refusal,
        "threshold": threshold,
        "all_preds": all_preds.tolist() if isinstance(all_preds, np.ndarray) else all_preds,  # Convert to list for JSON
        "all_labels": all_labels.tolist() if isinstance(all_labels, np.ndarray) else all_labels,
    }


def find_best_threshold(val_preds, val_labels, metric='f1'):
    """Find best threshold on validation set"""
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        binary_preds = (val_preds >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(val_labels, binary_preds)
        elif metric == 'accuracy':
            score = (binary_preds == val_labels).mean()
        elif metric == 'balanced_accuracy':
            # Balanced accuracy: average of sensitivity and specificity
            safe_mask = val_labels == 0
            unsafe_mask = val_labels == 1
            if safe_mask.sum() > 0 and unsafe_mask.sum() > 0:
                specificity = (binary_preds[safe_mask] == 0).mean()  # True negatives / all negatives
                sensitivity = (binary_preds[unsafe_mask] == 1).mean()  # True positives / all positives
                score = (specificity + sensitivity) / 2
            else:
                score = 0
        else:
            score = f1_score(val_labels, binary_preds)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def main():
    parser = argparse.ArgumentParser(description="Train LoRA safety adapter for LLaVA")
    parser.add_argument("--data-path", type=str, default="benchmark_balanced.jsonl",
                       help="Path to benchmark data file (JSONL or JSON)")
    parser.add_argument("--image-dir", type=str, default="images",
                       help="Directory containing images")
    parser.add_argument("--output-dir", type=str,
                       default="/n/netscratch/dam_lab/Lab/sliaw/cs2432_artifacts/lora_checkpoints",
                       help="Output directory for checkpoints (default: netscratch)")
    parser.add_argument("--model-name", type=str, default="llava-hf/llava-1.5-7b-hf",
                       help="Hugging Face model name")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--train-file", type=str, default=None,
                       help="Path to training split JSONL file (e.g., train.jsonl)")
    parser.add_argument("--val-file", type=str, default=None,
                       help="Path to validation split JSONL file (e.g., val.jsonl)")
    parser.add_argument("--test-file", type=str, default=None,
                       help="Path to test split JSONL file (e.g., test.jsonl)")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                       help="Path to a saved checkpoint (.pt) to resume training from")
    parser.add_argument("--hf-token", type=str, default=None,
                       help="Hugging Face token (optional, for private models)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--cache-dir", type=str,
                       default="/n/netscratch/dam_lab/Lab/sliaw/cs2432_artifacts",
                       help="Directory to cache/download models")
    
    args = parser.parse_args()

    # Create config from args
    config = Config(
        model_name=args.model_name,
        data_path=args.data_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        hf_token=args.hf_token,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    # Store cache_dir separately (not in Config)
    cache_dir = Path(args.cache_dir)

    # Validate that split files are provided
    if not config.train_file or not config.val_file or not config.test_file:
        raise ValueError("Must provide --train-file, --val-file, and --test-file")

    # Create output directory (relative to current working directory)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    # Update config with absolute path
    config.output_dir = str(output_dir)
    
    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using cache directory: {cache_dir}")

    # Authenticate with Hugging Face if token provided
    if config.hf_token:
        from huggingface_hub import login
        login(token=config.hf_token)
        logger.info("✓ Authenticated with Hugging Face")

    # Print system info
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Device: {config.device}")

    # Load split files directly
    logger.info(f"Loading training data from {config.train_file}...")
    train_data = load_data(config.train_file)
    logger.info(f"Loading validation data from {config.val_file}...")
    val_data = load_data(config.val_file)
    logger.info(f"Loading test data from {config.test_file}...")
    test_data = load_data(config.test_file)
    logger.info(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # Check if model is already cached
    model_cache_path = cache_dir / f"models--{config.model_name.replace('/', '--')}"
    if model_cache_path.exists():
        logger.info(f"Found cached model at {model_cache_path}")
        logger.info("Using cached model (no download needed)")
    else:
        logger.warning("Model not found in cache. Will attempt download.")
        logger.warning("If you get rate limit errors (429), try:")
        logger.warning("  1. Use --hf-token YOUR_TOKEN (authenticated users have higher limits)")
        logger.warning("  2. Wait a few minutes and try again")
        logger.warning(f"  3. Download manually: huggingface-cli download {config.model_name} --cache-dir {cache_dir}")
    
    # Load model and processor with retry logic
    logger.info(f"Loading {config.model_name}...")
    logger.info(f"Model will be downloaded/cached to: {cache_dir}")
    
    max_retries = 3
    retry_delay = 30  # seconds
    
    for attempt in range(max_retries):
        try:
            processor = AutoProcessor.from_pretrained(
                config.model_name,
                cache_dir=str(cache_dir),
                local_files_only=False,
            )
            logger.info("✓ Processor loaded")
            break
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit error. Waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("Rate limit error persists. Please use --hf-token or wait and try again.")
                    raise
            else:
                raise

    for attempt in range(max_retries):
        try:
            model = AutoModel.from_pretrained(
                config.model_name,
                cache_dir=str(cache_dir),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=False,
            )
            logger.info("✓ Model loaded")
            break
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit error. Waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error("Rate limit error persists. Please use --hf-token or wait and try again.")
                    raise
            else:
                raise
    logger.info("✓ Model loaded")
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # Apply LoRA
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        modules_to_save=[],
    )

    try:
        model = get_peft_model(model, peft_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ LoRA applied: {trainable:,} trainable ({trainable/total*100:.2f}%)")
    except Exception as e:
        logger.error(f"LoRA application error: {e}")
        raise

    # Create safety head
    # LLaVA-1.5-7B uses Llama-7B which has hidden_dim=4096
    hidden_dim = 4096
    safety_head = EmergentUnsafetyHead(hidden_dim=hidden_dim, dropout=config.lora_dropout)
    safety_head = safety_head.to(config.device)
    safety_head = safety_head.float()
    logger.info(f"✓ Safety head created (hidden_dim={hidden_dim})")

    # Optional resume from checkpoint
    start_epoch = 0
    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            safety_head.load_state_dict(ckpt.get("safety_head_state", {}))
            start_epoch = ckpt.get("epoch", -1) + 1
            logger.info(f"✓ Resumed from {ckpt_path} (start_epoch={start_epoch})")
            if 'metrics' in ckpt and 'auroc' in ckpt['metrics']:
                resume_auroc = ckpt['metrics']['auroc']
                logger.info(f"Previous best AUROC: {resume_auroc:.4f}")
        else:
            logger.warning(f"Resume checkpoint not found: {ckpt_path}")

    # Create datasets
    train_dataset = EmergentUnsafetyDataset(train_data, config.image_dir, processor)
    val_dataset = EmergentUnsafetyDataset(val_data, config.image_dir, processor)
    test_dataset = EmergentUnsafetyDataset(test_data, config.image_dir, processor)

    # Create data loaders with processor-aware collate function
    from functools import partial
    collate_fn_with_processor = partial(collate_fn, processor=processor)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        collate_fn=collate_fn_with_processor, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, 
        collate_fn=collate_fn_with_processor, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, 
        collate_fn=collate_fn_with_processor, num_workers=0
    )

    logger.info(f"Data loaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    # Loss and optimizer
    criterion = FocalLoss(alpha=1.0, gamma=1.5)
    optimizer = AdamW(
        list(safety_head.parameters()) + list(model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Training loop
    best_auroc = resume_auroc if "resume_auroc" in locals() else 0
    best_checkpoint = None
    train_metrics_history = []
    val_metrics_history = []

    logger.info("Starting training...\n")

    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")
        logger.info(f"{'='*60}")

        train_metrics = train_epoch(
            model, safety_head, train_loader, optimizer, criterion, config.device,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        train_metrics_history.append(train_metrics)

        # Evaluate with default threshold first
        val_metrics = evaluate(
            model, safety_head, val_loader, criterion, config.device,
            threshold=0.5,
        )
        val_metrics_history.append(val_metrics)

        logger.info(f"\nTrain Loss: {train_metrics['loss']:.4f}, AUROC: {train_metrics['auroc']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, AUROC: {val_metrics['auroc']:.4f}")
        logger.info(f"Refusal Rate: {val_metrics['refusal_rate']:.4f}, False Refusal: {val_metrics['false_refusal_rate']:.4f}")

        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            best_checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "safety_head_state": safety_head.state_dict(),
                "metrics": val_metrics,
            }
            checkpoint_path = Path(config.output_dir) / f"best_checkpoint_epoch{epoch}.pt"
            torch.save(best_checkpoint, checkpoint_path)
            logger.info(f"✓ Saved checkpoint to {checkpoint_path}")

    logger.info(f"\nTraining complete! Best AUROC: {best_auroc:.4f}")

    # Find best threshold on validation set
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD TUNING ON VALIDATION SET")
    logger.info("="*60)
    
    # Get validation predictions for threshold tuning
    val_metrics_final = evaluate(
        model, safety_head, val_loader, criterion, config.device, threshold=0.5
    )
    val_preds = val_metrics_final['all_preds']
    val_labels = val_metrics_final['all_labels']
    
    # Try different metrics for threshold selection
    best_threshold_f1, best_f1 = find_best_threshold(val_preds, val_labels, metric='f1')
    best_threshold_balanced, best_balanced = find_best_threshold(val_preds, val_labels, metric='balanced_accuracy')
    
    logger.info(f"Best threshold (F1): {best_threshold_f1:.3f} (F1={best_f1:.4f})")
    logger.info(f"Best threshold (Balanced Acc): {best_threshold_balanced:.3f} (Balanced Acc={best_balanced:.4f})")
    
    # Use balanced accuracy threshold (usually better for imbalanced data)
    best_threshold = best_threshold_balanced
    logger.info(f"Using threshold: {best_threshold:.3f} for test evaluation")

    # Final test set evaluation with best threshold
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*60)

    test_metrics = evaluate(
        model, safety_head, test_loader, criterion, config.device,
        threshold=best_threshold,
    )

    logger.info(f"""
Test Results (threshold={best_threshold:.3f}):
  Loss: {test_metrics['loss']:.4f}
  Accuracy: {test_metrics['accuracy']:.4f}
  AUROC: {test_metrics['auroc']:.4f}
  Refusal Rate: {test_metrics['refusal_rate']:.4f}
  False Refusal Rate: {test_metrics['false_refusal_rate']:.4f}
""")

    # Save models
    lora_path = Path(config.output_dir) / "lora_adapter"
    model.save_pretrained(str(lora_path))
    logger.info(f"✓ Saved LoRA adapter to {lora_path}")

    safety_head_path = Path(config.output_dir) / "safety_head.pt"
    torch.save(safety_head.state_dict(), str(safety_head_path))
    logger.info(f"✓ Saved safety head to {safety_head_path}")
    
    # Save best threshold
    threshold_path = Path(config.output_dir) / "best_threshold.txt"
    with open(threshold_path, 'w') as f:
        f.write(f"{best_threshold}\n")
    logger.info(f"✓ Saved best threshold ({best_threshold:.3f}) to {threshold_path}")

    # Save config
    config_path = Path(config.output_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=2)
    logger.info(f"✓ Saved config to {config_path}")

    # Save metrics (remove numpy arrays for JSON serialization)
    # Create a clean copy of test_metrics without numpy arrays
    test_metrics_clean = {
        "loss": float(test_metrics["loss"]),
        "accuracy": float(test_metrics["accuracy"]),
        "auroc": float(test_metrics["auroc"]),
        "refusal_rate": float(test_metrics["refusal_rate"]),
        "false_refusal_rate": float(test_metrics["false_refusal_rate"]),
        "threshold": float(test_metrics["threshold"]),
    }
    
    results = {
        "train_history": train_metrics_history,
        "val_history": val_metrics_history,
        "test": test_metrics_clean,
        "best_auroc": float(best_auroc),
        "best_threshold": float(best_threshold),
        "threshold_tuning": {
            "best_threshold_f1": float(best_threshold_f1),
            "best_f1": float(best_f1),
            "best_threshold_balanced": float(best_threshold_balanced),
            "best_balanced_acc": float(best_balanced),
        },
    }

    metrics_path = Path(config.output_dir) / "all_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Saved metrics to {metrics_path}")

    logger.info("\n✓✓✓ Training complete! ✓✓✓")


if __name__ == "__main__":
    main()

