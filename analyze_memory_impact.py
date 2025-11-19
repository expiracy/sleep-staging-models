"""
Analyze potential memory reduction strategies for windowed sleep staging model.
Compares different optimization approaches and their expected memory impact.
"""

import torch
import torch.nn as nn
import argparse
import json
from pathlib import Path
import sys
from models.multimodal_model_crossattn import ImprovedMultiModalSleepNet
from models.multimodal_model_crossattn_windowed import WindowAdaptiveSleepNet
import torch_load_utils


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_activation_memory(model, batch_size=1, seq_length=2400, d_model=256):
    """
    Estimate activation memory for cross-attention blocks.
    
    Attention memory = batch_size * n_heads * seq_length^2 * 4 bytes (FP32)
    For cross-attention: seq_length_ppg * seq_length_ecg
    """
    n_heads = model.cross_attention[0].cross_attention.n_heads if hasattr(model, 'cross_attention') else 8
    n_blocks = len(model.cross_attention) if hasattr(model, 'cross_attention') else 3
    
    # Bidirectional cross-attention: PPG->ECG and ECG->PPG
    attention_maps_per_block = 2
    
    # Each attention map: [batch, n_heads, seq_len, seq_len]
    single_attention_bytes = batch_size * n_heads * seq_length * seq_length * 4
    total_attention_bytes = single_attention_bytes * attention_maps_per_block * n_blocks
    
    # Feature maps: [batch, seq_len, d_model]
    feature_bytes_per_stream = batch_size * seq_length * d_model * 4
    total_feature_bytes = feature_bytes_per_stream * 2 * (n_blocks + 1)  # PPG + ECG, input + each block
    
    # Residual connections and intermediate layers
    intermediate_bytes = feature_bytes_per_stream * 4 * n_blocks  # Conv blocks
    
    return {
        'attention_maps_mb': total_attention_bytes / (1024**2),
        'feature_maps_mb': total_feature_bytes / (1024**2),
        'intermediate_mb': intermediate_bytes / (1024**2),
        'total_mb': (total_attention_bytes + total_feature_bytes + intermediate_bytes) / (1024**2)
    }


def analyze_optimization_strategies(model, seq_length=2400):
    """
    Analyze different optimization strategies and their memory impact.
    """
    total_params, trainable_params = count_parameters(model)
    param_memory_mb = (total_params * 4) / (1024**2)  # FP32
    
    activation_memory = estimate_activation_memory(model, batch_size=1, seq_length=seq_length)
    
    strategies = {}
    
    # 1. Weight Pruning (from sparsity analysis: 92.77% near-zero)
    # This prunes ATTENTION WEIGHTS (computed), not model parameters
    strategies['attention_weight_pruning'] = {
        'description': 'Prune 93% of computed attention weights (make them exactly zero)',
        'speedup': '14x attention computation',
        'memory_reduction_mb': 0,  # Attention weights are computed, not stored
        'memory_reduction_percent': 0,
        'implementation': 'Mask attention matrix, use sparse matmul',
        'notes': 'Reduces compute/FLOPs but NOT memory unless using sparse tensors'
    }
    
    # 2. Model Parameter Pruning
    # Target: Conv layers, linear layers in fusion blocks
    strategies['model_parameter_pruning'] = {
        'description': 'Prune 50% of model parameters (weights & biases)',
        'speedup': '1.5-2x inference',
        'memory_reduction_mb': param_memory_mb * 0.5,
        'memory_reduction_percent': (param_memory_mb * 0.5) / activation_memory['total_mb'] * 100,
        'implementation': 'PyTorch prune module, then convert to sparse format',
        'notes': 'Parameters are only ~7% of total memory, limited impact'
    }
    
    # 3. Quantization (FP32 -> INT8)
    strategies['int8_quantization'] = {
        'description': 'Quantize model to INT8 (4x reduction in weight size)',
        'speedup': '2-4x inference with INT8 ops',
        'memory_reduction_mb': param_memory_mb * 0.75,  # 4x smaller
        'memory_reduction_percent': (param_memory_mb * 0.75) / activation_memory['total_mb'] * 100,
        'implementation': 'torch.quantization.quantize_dynamic or ONNX INT8',
        'notes': 'Best for parameter memory, activations still FP32 during compute'
    }
    
    # 4. Low-Rank Factorization
    # From sparsity: effective rank 139/240 (58%) -> can reduce to ~100
    strategies['low_rank_factorization'] = {
        'description': 'Factor attention QKV projections: d_model=256 -> rank=100',
        'speedup': '1.5-2x attention computation',
        'memory_reduction_mb': 0,  # Projections are small part of model
        'memory_reduction_percent': 0,
        'implementation': 'Replace nn.Linear(d_model, d_model) with two low-rank layers',
        'notes': 'Reduces FLOPs significantly, minimal memory impact'
    }
    
    # 5. Reduce Sequence Length (chunked inference)
    # Already tried - user reverted
    strategies['chunked_inference'] = {
        'description': 'Process 600 samples at a time instead of 2400',
        'speedup': '1x (no change)',
        'memory_reduction_mb': activation_memory['attention_maps_mb'] * 0.75,  # Quadratic reduction
        'memory_reduction_percent': (activation_memory['attention_maps_mb'] * 0.75) / activation_memory['total_mb'] * 100,
        'implementation': 'Already implemented but user reverted',
        'notes': 'Reduces activation memory but requires boundary handling'
    }
    
    # 6. Gradient Checkpointing (training only)
    strategies['gradient_checkpointing'] = {
        'description': 'Trade compute for memory by recomputing activations',
        'speedup': '0.5-0.7x (slower)',
        'memory_reduction_mb': activation_memory['intermediate_mb'] * 0.5,
        'memory_reduction_percent': (activation_memory['intermediate_mb'] * 0.5) / activation_memory['total_mb'] * 100,
        'implementation': 'torch.utils.checkpoint.checkpoint()',
        'notes': 'Only helps during training, not inference'
    }
    
    # 7. Reduce Model Dimensions
    strategies['reduce_dimensions'] = {
        'description': 'Reduce d_model from 256->128, n_heads from 8->4',
        'speedup': '2-3x inference',
        'memory_reduction_mb': param_memory_mb * 0.6 + activation_memory['total_mb'] * 0.5,
        'memory_reduction_percent': 50,
        'implementation': 'Retrain model with smaller architecture',
        'notes': 'Most effective but requires retraining, may hurt accuracy'
    }
    
    # 8. Flash Attention / Memory-Efficient Attention
    strategies['flash_attention'] = {
        'description': 'Use memory-efficient attention implementation (xformers)',
        'speedup': '2-3x attention computation',
        'memory_reduction_mb': activation_memory['attention_maps_mb'] * 0.6,
        'memory_reduction_percent': (activation_memory['attention_maps_mb'] * 0.6) / activation_memory['total_mb'] * 100,
        'implementation': 'xformers.ops.memory_efficient_attention or torch.nn.functional.scaled_dot_product_attention',
        'notes': 'Reduces peak memory during attention, no accuracy loss'
    }
    
    return {
        'model_info': {
            'total_parameters': total_params,
            'parameter_memory_mb': param_memory_mb,
            'estimated_activation_memory': activation_memory,
            'total_estimated_memory_mb': param_memory_mb + activation_memory['total_mb']
        },
        'optimization_strategies': strategies
    }


def print_analysis(analysis):
    """Print formatted analysis results."""
    print("=" * 80)
    print("MEMORY OPTIMIZATION ANALYSIS")
    print("=" * 80)
    
    info = analysis['model_info']
    print(f"\nModel Information:")
    print(f"  Total Parameters: {info['total_parameters']:,}")
    print(f"  Parameter Memory: {info['parameter_memory_mb']:.2f} MB")
    print(f"\nEstimated Activation Memory (batch=1, seq=2400):")
    act = info['estimated_activation_memory']
    print(f"  Attention Maps: {act['attention_maps_mb']:.2f} MB")
    print(f"  Feature Maps: {act['feature_maps_mb']:.2f} MB")
    print(f"  Intermediate: {act['intermediate_mb']:.2f} MB")
    print(f"  Total Activations: {act['total_mb']:.2f} MB")
    print(f"\nTotal Estimated Memory: {info['total_estimated_memory_mb']:.2f} MB")
    print(f"\n{'='*80}")
    
    print("\nOPTIMIZATION STRATEGIES (Ranked by Memory Impact):")
    print("=" * 80)
    
    # Sort by memory reduction
    strategies = analysis['optimization_strategies']
    sorted_strategies = sorted(
        strategies.items(),
        key=lambda x: x[1]['memory_reduction_mb'],
        reverse=True
    )
    
    for i, (name, strategy) in enumerate(sorted_strategies, 1):
        print(f"\n{i}. {name.upper().replace('_', ' ')}")
        print(f"   Description: {strategy['description']}")
        print(f"   Memory Reduction: {strategy['memory_reduction_mb']:.2f} MB ({strategy['memory_reduction_percent']:.1f}%)")
        print(f"   Speedup: {strategy['speedup']}")
        print(f"   Implementation: {strategy['implementation']}")
        print(f"   Notes: {strategy['notes']}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("\nFor MAXIMUM memory reduction during inference:")
    print("  1. Flash Attention (30-40% activation memory reduction, NO retraining)")
    print("  2. INT8 Quantization (75% parameter memory reduction)")
    print("  3. Reduce model dimensions (50% total reduction, requires retraining)")
    
    print("\nFor SPEED without memory reduction:")
    print("  1. Attention weight pruning (14x attention speedup from your sparsity analysis)")
    print("  2. Low-rank factorization (1.5-2x speedup, minimal accuracy loss)")
    
    print("\nNOTE: Your sparsity analysis (92.77% near-zero attention weights) is excellent")
    print("      for SPEED optimization but does NOT directly reduce memory usage.")
    print("      Attention weights are computed on-the-fly, not stored in memory.")


def main():
    parser = argparse.ArgumentParser(description='Analyze memory optimization strategies')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='memory_analysis.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch_load_utils.safe_torch_load(args.checkpoint, map_location='cpu')
    
    # Detect model type
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    is_windowed = any('window_encoder' in k for k in state_dict.keys())
    
    if is_windowed:
        print("Detected WindowAdaptiveSleepNet model")
        model_config = checkpoint.get('config', {}).get('model', {})
        model = WindowAdaptiveSleepNet(
            n_classes=model_config.get('n_classes', 4),
            d_model=model_config.get('d_model', 256),
            n_heads=model_config.get('n_heads', 8),
            n_fusion_blocks=model_config.get('n_fusion_blocks', 3),
            dropout=model_config.get('dropout', 0.1)
        )
        seq_length = model_config.get('max_windows', 11) * 240  # After window encoding
    else:
        print("Detected ImprovedMultiModalSleepNet model")
        model_config = checkpoint.get('config', {}).get('model', {})
        model = ImprovedMultiModalSleepNet(
            n_classes=model_config.get('n_classes', 4),
            d_model=model_config.get('d_model', 256),
            n_heads=model_config.get('n_heads', 8),
            n_fusion_blocks=model_config.get('n_fusion_blocks', 3)
        )
        seq_length = 2400  # After PPG/ECG encoders
    
    model.load_state_dict(state_dict)
    
    # Analyze optimization strategies
    print("\nAnalyzing optimization strategies...")
    analysis = analyze_optimization_strategies(model, seq_length=seq_length)
    
    # Print results
    print_analysis(analysis)
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    def make_serializable(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        else:
            return str(obj)
    
    with open(output_path, 'w') as f:
        json.dump(make_serializable(analysis), f, indent=2)
    
    print(f"\nAnalysis saved to {output_path}")


if __name__ == '__main__':
    main()
