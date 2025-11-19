# Window-Adaptive Sleep Staging Model

## Overview

This implementation enables the sleep staging model to work with **variable-length input sequences**, from short 5-minute windows up to full 10-hour recordings. This is achieved through:

1. **Sequence-length agnostic architecture**
2. **Learned positional encodings** (supports any length)
3. **Curriculum learning** (train on large windows first, then smaller)
4. **Adaptive interpolation** to match input/output lengths

## Key Changes from Original Model

### 1. Variable-Length Support

**Before (fixed 1200 epochs):**
```python
output_features = F.avg_pool1d(refined_features, kernel_size=2, stride=2)
if output_features.size(2) != 1200:
    output_features = F.interpolate(output_features, size=1200, ...)
```

**After (dynamic):**
```python
n_epochs = ppg.size(2) // 1024  # Calculate from input
output_features = F.interpolate(refined_features, size=n_epochs, ...)
```

### 2. Learned Positional Encoding

**Before (fixed-size positional encoding):**
```python
self.positional_encoding = nn.Parameter(torch.randn(1, d_model, 2400))
```

**After (learnable, interpolates to any size):**
```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, max_len))
    
    def forward(self, x):
        seq_len = x.shape[2]
        if seq_len > self.max_len:
            pos_enc = F.interpolate(self.pos_embedding, size=seq_len, ...)
        else:
            pos_enc = self.pos_embedding[:, :, :seq_len]
        return x + pos_enc
```

### 3. Fixed Temporal Convolutions

**Before:** Used `Chomp1d` which caused size mismatches

**After:** Use proper 'same' padding
```python
padding = (kernel_size - 1) * dilation // 2
self.conv1 = nn.Conv1d(..., padding=padding, dilation=dilation)
```

## Architecture

```
Input: (B, 1, variable_samples)
  ↓
Encoder (9 ResConvBlocks): ÷512
  → (B, d_model, samples//512)
  ↓
Positional Encoding (learned, adaptive)
  ↓
Modality Weighting
  ↓
Cross-Modal Fusion (3 blocks)
  ↓
Temporal Modeling (dilated convs)
  ↓
Adaptive Interpolation: → n_epochs
  ↓
Classifier
  ↓
Output: (B, 4, n_epochs)
```

## Curriculum Learning Strategy

Training progressively introduces smaller windows to help the model learn both global and local patterns:

### Stage 1: Large Windows (Epochs 1-15)
- Window sizes: 150-240 epochs (75-120 minutes)
- **Purpose:** Learn global sleep architecture patterns
- Model sees large temporal context

### Stage 2: Medium-Large Windows (Epochs 16-30)
- Window sizes: 80-150 epochs (40-75 minutes)
- **Purpose:** Transition to more localized patterns

### Stage 3: Medium Windows (Epochs 31-45)
- Window sizes: 40-100 epochs (20-50 minutes)
- **Purpose:** Learn local sleep stage transitions

### Stage 4: Mixed Windows (Epochs 46-60)
- Window sizes: 20-150 epochs (10-75 minutes, mixed)
- **Purpose:** Fine-tune on diverse window sizes
- Model becomes truly adaptive

## Training

```powershell
cd src\sleep_staging_models

# Train window-adaptive model
python train_windowed_crossattn.py \
  --config configs/config_windowed_adaptive.yaml \
  --output_dir ../../outputs/windowed_adaptive_run1
```

### Training Configuration

```yaml
training:
  batch_size: 4
  learning_rate: 0.0001
  use_amp: true
  
  curriculum:
    stage1:
      epochs: 15
      window_sizes: [150, 180, 200, 240]
    # ... more stages
```

## Testing Variable Lengths

```powershell
# Test that model works with different input sizes
python models/multimodal_model_crossattn_windowed.py
```

Expected output:
```
10 epochs (5 min):    ✓
20 epochs (10 min):   ✓
60 epochs (30 min):   ✓
120 epochs (1 hour):  ✓
1200 epochs (10 hrs): ✓
```

## Inference

After training, the model can handle any window size:

```python
from models.multimodal_model_crossattn_windowed import WindowAdaptiveSleepNet

model = WindowAdaptiveSleepNet()
model.load_state_dict(checkpoint['model_state_dict'])

# Works with any length!
ppg_5min = torch.randn(1, 1, 10240)    # 10 epochs
ppg_1hr = torch.randn(1, 1, 122880)   # 120 epochs
ppg_full = torch.randn(1, 1, 1228800) # 1200 epochs

out_5min = model(ppg_5min, ecg_5min)   # → (1, 4, 10)
out_1hr = model(ppg_1hr, ecg_1hr)      # → (1, 4, 120)
out_full = model(ppg_full, ecg_full)   # → (1, 4, 1200)
```

## Expected Performance

### Training Time
- **Total epochs:** 60 (across 4 curriculum stages)
- **Time per epoch:** ~20-30 minutes (depends on window sampling)
- **Total training:** ~20-30 hours on single GPU

### Expected Accuracy
After curriculum learning, the model should achieve:
- **Large windows (60+ min):** 90-95% of baseline performance
- **Medium windows (20-60 min):** 85-90% of baseline
- **Small windows (10-20 min):** 80-85% of baseline

The model learns to balance:
- **Global context** (from large window training)
- **Local patterns** (from small window training)

## Memory Savings

| Window Size | Samples | Memory vs Full | Can Use Batch Size |
|-------------|---------|----------------|-------------------|
| 10 epochs   | 10,240  | ~120x less    | 16-32            |
| 30 epochs   | 30,720  | ~40x less     | 8-16             |
| 60 epochs   | 61,440  | ~20x less     | 4-8              |
| 120 epochs  | 122,880 | ~10x less     | 2-4              |
| 1200 epochs | 1,228,800| 1x (baseline) | 1-2              |

## Files

- `models/multimodal_model_crossattn_windowed.py` - Window-adaptive model architecture
- `train_windowed_crossattn.py` - Training script with curriculum learning
- `configs/config_windowed_adaptive.yaml` - Training configuration
- `windowed_dataset.py` - Dataset for windowed training (from earlier)
- `windowed_inference.py` - Inference with trained model (from earlier)

## Advantages

✅ Works with any sequence length (10 to 1200 epochs)
✅ Memory-efficient training and inference  
✅ Enables streaming/real-time applications
✅ Learns both global and local temporal patterns
✅ Can process partial recordings

## Limitations

⚠️ Requires retraining from scratch
⚠️ May not match full-sequence model performance initially
⚠️ More complex training procedure (curriculum learning)
⚠️ Longer total training time

## Next Steps

1. **Train the model** with curriculum learning
2. **Compare performance** at different window sizes
3. **Fine-tune curriculum** if needed (adjust stage durations)
4. **Test on streaming data** for real-time inference

## Comparison with Original

| Aspect | Original Model | Window-Adaptive Model |
|--------|---------------|----------------------|
| Input size | Fixed 1200 epochs | Variable (10-1200 epochs) |
| Training | Simple, one configuration | Curriculum learning |
| Memory | High (full sequence) | Low-High (depends on window) |
| Inference | Batch the full recording | Can stream in windows |
| Parameters | 14.9M | 12.3M (slightly smaller) |
| Use case | Offline batch processing | Streaming + batch |
