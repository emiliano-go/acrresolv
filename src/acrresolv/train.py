"""Training pipeline for T5-small + LoRA acronym expansion model.

Uses quantization-aware training (QAT) with fake quantization for W4A8
quantization, LoRA fine-tuning, and exports to .acr format.
"""
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW


# ============================================================================
# Configuration
# ============================================================================

_DATA_PATH = Path(__file__).parent / "data.json"
_MODEL_DIR = Path(__file__).parent / "models"


# ============================================================================
# Dataset
# ============================================================================

class AcronymDataset(Dataset):
    """Dataset for acronym expansion training."""
    
    def __init__(self, data: list[tuple[str, str]], tokenizer,
                 max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        phrase, acronym = self.data[idx]
        
        # Format input: "central processing unit"
        input_text = phrase
        
        # Format target: "CPU"
        target_text = acronym
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        # Create labels (ignore padding tokens)
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels,
        }


# ============================================================================
# Training Functions
# ============================================================================

def train(
    data: list[tuple[str, str]],
    model_name: str = 't5-small',
    epochs: int = 10,
    lr: float = 3e-4,
    batch_size: int = 16,
    val_split: float = 0.1,
    patience: int = 5,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    use_qat: bool = True,
    weight_bits: int = 4,
    group_size: int = 128,
    max_length: int = 128,
    device: Optional[str] = None,
) -> T5ForConditionalGeneration:
    """Train T5 model with LoRA and QAT.
    
    Args:
        data: List of (phrase, acronym) pairs
        model_name: HuggingFace model name
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        val_split: Validation split ratio
        patience: Early stopping patience
        use_lora: Whether to use LoRA
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        use_qat: Whether to use quantization-aware training
        weight_bits: Weight quantization bits
        group_size: Quantization group size
        max_length: Maximum sequence length
        device: Device to train on
    
    Returns:
        Trained model
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set seeds
    random.seed(42)
    torch.manual_seed(42)
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Load base model
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    
    # Apply QAT if requested (fake quant during training)
    if use_qat:
        from .quantize import QATHook
        qat_hook = QATHook(model, bits=weight_bits, group_size=group_size)
        qat_hook.register()
        print(f"QAT enabled: W{weight_bits}A8")
    
    # Prepare data
    random.shuffle(data)
    split = int(len(data) * (1 - val_split))
    train_data, val_data = data[:split], data[split:]
    
    train_dataset = AcronymDataset(train_data, tokenizer, max_length)
    val_dataset = AcronymDataset(val_data, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Training loop
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if (step + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{epochs}  step {step+1}/{len(train_loader)}  loss {loss.item():.4f}", flush=True)
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / max(len(val_loader), 1)
        
        # Print progress
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={avg_val_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Save model
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as HuggingFace format
    model_path = _MODEL_DIR / "t5_acronym"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"Model saved to {model_path}")
    
    return model


def load_model(path: Optional[str] = None, device: str = 'cpu') -> T5ForConditionalGeneration:
    """Load trained model.
    
    Args:
        path: Model path (default: models/t5_acronym)
        device: Device to load to
    
    Returns:
        Loaded model
    """
    if path is None:
        path = _MODEL_DIR / "t5_acronym"
    
    model = T5ForConditionalGeneration.from_pretrained(path)
    model = model.to(device)
    model.eval()
    
    return model


def predict(model: T5ForConditionalGeneration, tokenizer, text: str,
            max_length: int = 128) -> str:
    """Predict acronym from phrase.
    
    Args:
        model: T5 model
        tokenizer: Tokenizer
        text: Input phrase
        max_length: Maximum output length
    
    Returns:
        Predicted acronym
    """
    input_ids = tokenizer(text, return_tensors='pt').input_ids
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_synthetic_data(existing_data: list[tuple[str, str]],
                           num_samples: int = 1000) -> list[tuple[str, str]]:
    """Generate synthetic training data.
    
    Creates variations of existing phrase-acronym pairs by:
    - Changing word order
    - Adding/removing words
    - Using synonyms
    - Different capitalization
    
    Args:
        existing_data: Existing phrase-acronym pairs
        num_samples: Number of synthetic samples to generate
    
    Returns:
        List of (phrase, acronym) pairs
    """
    synthetic = []
    
    # Patterns for generating variations
    patterns = [
        # First letter of each word
        lambda words: ''.join(w[0].upper() for w in words if w),
        # First two letters
        lambda words: ''.join(w[:2].upper() for w in words if w)[:6],
        # First letter + last letter
        lambda words: ''.join(w[0].upper() for w in words if w)[:4],
    ]
    
    for phrase, acronym in existing_data:
        words = phrase.split()
        
        if len(words) < 2:
            continue
        
        # Generate variations
        for _ in range(min(3, num_samples // len(existing_data))):
            # Randomly select a pattern
            pattern = random.choice(patterns)
            
            # Shuffle words
            shuffled = words.copy()
            random.shuffle(shuffled)
            
            # Apply pattern
            new_acronym = pattern(shuffled)
            
            if new_acronym and new_acronym != acronym:
                synthetic.append((phrase, new_acronym))
        
        if len(synthetic) >= num_samples:
            break
    
    return synthetic[:num_samples]


# ============================================================================
# Main Training Script
# ============================================================================

def _build_and_train():
    """Main training function."""
    # Load data
    with open(_DATA_PATH) as f:
        data = [tuple(pair) for pair in json.load(f)]
    
    print(f"Training on {len(data)} examples...")
    
    # Train model
    model = train(
        data,
        epochs=20,
        lr=5e-4,
        batch_size=8,
        val_split=0.1,
        patience=5,
        use_lora=False,
        use_qat=False,
        weight_bits=4,
        device='cuda',
        model_name='google-t5/t5-small',
    )
    
    # Quantize and save
    from .quantize import save_quantized_model
    quant_path = _MODEL_DIR / "t5_w4"
    save_quantized_model(model, T5Tokenizer.from_pretrained('google-t5/t5-small'),
                         str(quant_path), bits=4, group_size=128)
    
    # Evaluate hybrid system
    from .resolver import resolve
    with open(_DATA_PATH) as f:
        eval_data = [tuple(pair) for pair in json.load(f)]
    
    correct = sum(1 for p, a in eval_data if resolve(p).upper() == a.upper())
    print(f"\nHybrid training accuracy: {correct}/{len(eval_data)} ({100*correct/len(eval_data):.1f}%)")


if __name__ == "__main__":
    _build_and_train()
