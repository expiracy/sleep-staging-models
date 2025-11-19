"""
Attention Sparsity Analysis

Analyzes whether attention patterns are sparse (concentrated) or dense (distributed):
1. Entropy analysis - measures attention distribution
2. Top-K sparsity - how much attention goes to top K positions
3. Gini coefficient - inequality in attention distribution
4. Effective rank - dimensionality of attention
5. L1/L2 norms - mathematical sparsity measures

This helps determine if sparse attention mechanisms could replace full attention.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import h5py
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from models.multimodal_model_crossattn_windowed import WindowAdaptiveSleepNet


def calculate_entropy(attention_weights):
    """
    Calculate Shannon entropy of attention distribution
    
    High entropy = distributed attention
    Low entropy = concentrated attention
    """
    # Flatten to 1D if needed
    if attention_weights.ndim > 1:
        attention_weights = attention_weights.flatten()
    
    # Add small epsilon to avoid log(0)
    attention_weights = attention_weights + 1e-10
    attention_weights = attention_weights / attention_weights.sum()
    
    entropy = -np.sum(attention_weights * np.log2(attention_weights))
    
    # Normalize by max possible entropy
    max_entropy = np.log2(len(attention_weights))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return entropy, normalized_entropy


def calculate_top_k_mass(attention_weights, k_values=[1, 5, 10, 20, 50]):
    """
    Calculate what percentage of total attention goes to top-K positions
    
    High top-K mass = sparse (concentrated)
    Low top-K mass = dense (distributed)
    """
    # Flatten if needed
    if attention_weights.ndim > 1:
        attention_weights = attention_weights.flatten()
    
    # Sort descending
    sorted_weights = np.sort(attention_weights)[::-1]
    
    results = {}
    for k in k_values:
        if k <= len(sorted_weights):
            top_k_sum = np.sum(sorted_weights[:k])
            total_sum = np.sum(sorted_weights)
            results[k] = top_k_sum / total_sum if total_sum > 0 else 0
    
    return results


def calculate_gini_coefficient(attention_weights):
    """
    Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    
    High Gini = sparse (unequal distribution)
    Low Gini = dense (equal distribution)
    """
    # Flatten if needed
    if attention_weights.ndim > 1:
        attention_weights = attention_weights.flatten()
    
    # Sort
    sorted_weights = np.sort(attention_weights)
    n = len(sorted_weights)
    
    # Calculate Gini
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
    
    return gini


def calculate_effective_rank(attention_weights):
    """
    Calculate effective rank using entropy
    
    Low rank = sparse (attention focuses on few dimensions)
    High rank = dense (attention uses many dimensions)
    """
    # Flatten if needed
    if attention_weights.ndim > 1:
        attention_weights = attention_weights.flatten()
    
    # Normalize
    attention_weights = attention_weights / (np.sum(attention_weights) + 1e-10)
    
    # Calculate entropy
    entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10))
    
    # Effective rank
    effective_rank = np.exp(entropy)
    
    # Normalize by length
    normalized_rank = effective_rank / len(attention_weights)
    
    return effective_rank, normalized_rank


def calculate_sparsity_ratio(attention_weights, threshold=0.01):
    """
    Calculate sparsity ratio: fraction of weights below threshold
    
    High ratio = sparse (many near-zero weights)
    Low ratio = dense (few near-zero weights)
    """
    if attention_weights.ndim > 1:
        attention_weights = attention_weights.flatten()
    
    below_threshold = np.sum(attention_weights < threshold)
    sparsity_ratio = below_threshold / len(attention_weights)
    
    return sparsity_ratio


def calculate_l1_l2_ratio(attention_weights):
    """
    Calculate L1/L2 norm ratio (Hoyer sparsity)
    
    Values closer to 1 = more sparse
    Values closer to 0 = more dense
    
    Formula: (sqrt(n) - L1/L2) / (sqrt(n) - 1)
    where n is the number of elements
    """
    if attention_weights.ndim > 1:
        attention_weights = attention_weights.flatten()
    
    n = len(attention_weights)
    l1_norm = np.sum(np.abs(attention_weights))
    l2_norm = np.sqrt(np.sum(attention_weights ** 2))
    
    if l2_norm > 0:
        ratio = l1_norm / l2_norm
        # Hoyer sparsity measure
        hoyer = (np.sqrt(n) - ratio) / (np.sqrt(n) - 1) if n > 1 else 0
    else:
        ratio = 0
        hoyer = 0
    
    return ratio, hoyer


class AttentionSparsityAnalyzer:
    """Analyze sparsity of attention patterns"""
    
    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        if config_path is None:
            config_path = Path(checkpoint_path).parent / 'config.json'
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"Loading windowed model from {checkpoint_path}")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Storage
        self.attention_weights = {}
        self._register_hooks()
    
    def _load_model(self):
        """Load the windowed model"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        model_params = self.config.get('model', {})
        d_model = model_params.get('d_model', 256)
        n_heads = model_params.get('n_heads', 8)
        n_fusion_blocks = model_params.get('n_fusion_blocks', 3)
        
        model = WindowAdaptiveSleepNet(
            n_classes=4,
            d_model=d_model,
            n_heads=n_heads,
            n_fusion_blocks=n_fusion_blocks
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Model loaded: d_model={d_model}, n_heads={n_heads}, n_fusion_blocks={n_fusion_blocks}")
        
        return model
    
    def _register_hooks(self):
        """Register hooks to capture attention"""
        self.hooks = []
        
        for idx, fusion_block in enumerate(self.model.fusion_blocks):
            def make_ppg_hook(block_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) == 2:
                        self.attention_weights[f'block_{block_idx}_ppg'] = output[1].detach().cpu()
                return hook
            
            def make_ecg_hook(block_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) == 2:
                        self.attention_weights[f'block_{block_idx}_ecg'] = output[1].detach().cpu()
                return hook
            
            self.hooks.append(fusion_block.ppg_cross_attn.register_forward_hook(make_ppg_hook(idx)))
            self.hooks.append(fusion_block.ecg_cross_attn.register_forward_hook(make_ecg_hook(idx)))
    
    def analyze_window(self, ppg_data, ecg_data):
        """Analyze sparsity for a single window"""
        self.attention_weights = {}
        
        ppg_tensor = torch.FloatTensor(ppg_data).unsqueeze(0).unsqueeze(0).to(self.device)
        ecg_tensor = torch.FloatTensor(ecg_data).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _ = self.model(ppg_tensor, ecg_tensor)
        
        return self.attention_weights.copy()
    
    def analyze_subject_sparsity(self, h5_ppg_path, h5_ecg_path, h5_index_path, subject_id, n_windows=None):
        """Comprehensive sparsity analysis for a subject"""
        print(f"\nAnalyzing attention sparsity for subject {subject_id}")
        
        # Load data
        with h5py.File(h5_index_path, 'r') as f:
            subject_key = f'{subject_id:04d}'
            window_indices = f[f'subjects/{subject_key}/window_indices'][:]
            start_idx = window_indices[0]
            n_epochs = len(window_indices)
        
        with h5py.File(h5_ppg_path, 'r') as f:
            ppg_epochs = f['ppg'][start_idx:start_idx + n_epochs]
        
        with h5py.File(h5_ecg_path, 'r') as f:
            ecg_epochs = f['ecg'][start_idx:start_idx + n_epochs]
        
        # Window configuration
        window_epochs = self.config['windowing']['window_epochs']
        overlap_percent = self.config['windowing']['overlap_percent']
        
        samples_per_epoch = 1024
        chunk_size = window_epochs * samples_per_epoch
        overlap_epochs = int(window_epochs * overlap_percent / 100)
        stride = (window_epochs - overlap_epochs) * samples_per_epoch
        
        # Collect sparsity metrics
        all_metrics = {
            'entropy': [],
            'normalized_entropy': [],
            'top_k_mass': {k: [] for k in [1, 5, 10, 20, 50]},
            'gini': [],
            'effective_rank': [],
            'normalized_rank': [],
            'sparsity_ratio': [],
            'l1_l2_ratio': [],
            'hoyer_sparsity': []
        }
        
        ppg_continuous = ppg_epochs.reshape(-1)
        ecg_continuous = ecg_epochs.reshape(-1)
        
        window_count = 0
        for start_idx in tqdm(range(0, len(ppg_continuous) - chunk_size + 1, stride), desc="Analyzing sparsity"):
            ppg_window = ppg_continuous[start_idx:start_idx + chunk_size]
            ecg_window = ecg_continuous[start_idx:start_idx + chunk_size]
            
            attention_weights = self.analyze_window(ppg_window, ecg_window)
            
            # Analyze each attention layer
            for key, attn in attention_weights.items():
                # attn shape: (batch=1, n_heads, seq_len, seq_len)
                # Average over batch and heads
                attn_avg = attn[0].mean(dim=0).numpy()  # (seq_len, seq_len)
                
                # Average over query dimension to get attention distribution
                attn_dist = attn_avg.mean(axis=0)  # (seq_len,)
                
                # Calculate metrics
                entropy, norm_entropy = calculate_entropy(attn_dist)
                all_metrics['entropy'].append(entropy)
                all_metrics['normalized_entropy'].append(norm_entropy)
                
                top_k = calculate_top_k_mass(attn_dist)
                for k, mass in top_k.items():
                    all_metrics['top_k_mass'][k].append(mass)
                
                gini = calculate_gini_coefficient(attn_dist)
                all_metrics['gini'].append(gini)
                
                eff_rank, norm_rank = calculate_effective_rank(attn_dist)
                all_metrics['effective_rank'].append(eff_rank)
                all_metrics['normalized_rank'].append(norm_rank)
                
                sparsity = calculate_sparsity_ratio(attn_dist)
                all_metrics['sparsity_ratio'].append(sparsity)
                
                l1_l2, hoyer = calculate_l1_l2_ratio(attn_dist)
                all_metrics['l1_l2_ratio'].append(l1_l2)
                all_metrics['hoyer_sparsity'].append(hoyer)
            
            window_count += 1
            if n_windows and window_count >= n_windows:
                break
        
        return all_metrics
    
    def print_sparsity_report(self, metrics):
        """Print comprehensive sparsity report"""
        print("\n" + "="*70)
        print("ATTENTION SPARSITY ANALYSIS REPORT")
        print("="*70)
        
        print("\n1. ENTROPY ANALYSIS (Lower = More Sparse)")
        print(f"   Normalized Entropy: {np.mean(metrics['normalized_entropy']):.4f} ± {np.std(metrics['normalized_entropy']):.4f}")
        print(f"   Range: [{np.min(metrics['normalized_entropy']):.4f}, {np.max(metrics['normalized_entropy']):.4f}]")
        
        if np.mean(metrics['normalized_entropy']) < 0.5:
            print("   → Very sparse attention! Only focusing on ~50% of positions")
        elif np.mean(metrics['normalized_entropy']) < 0.7:
            print("   → Moderately sparse attention")
        else:
            print("   → Dense attention - utilizing most positions")
        
        print("\n2. TOP-K CONCENTRATION (Higher = More Sparse)")
        for k in [1, 5, 10, 20, 50]:
            if k in metrics['top_k_mass']:
                avg_mass = np.mean(metrics['top_k_mass'][k])
                print(f"   Top-{k:2d} positions capture: {avg_mass:.2%} of attention")
        
        top_10_mass = np.mean(metrics['top_k_mass'][10])
        if top_10_mass > 0.5:
            print(f"   → Highly sparse! Top-10 positions get >50% attention")
            print(f"   → Candidate for Top-K sparse attention (K≈10-20)")
        elif top_10_mass > 0.3:
            print(f"   → Moderate sparsity - could benefit from pruning")
        
        print("\n3. GINI COEFFICIENT (Higher = More Sparse)")
        print(f"   Gini: {np.mean(metrics['gini']):.4f} ± {np.std(metrics['gini']):.4f}")
        
        if np.mean(metrics['gini']) > 0.8:
            print("   → Highly unequal distribution - very sparse!")
        elif np.mean(metrics['gini']) > 0.6:
            print("   → Moderate inequality - some sparsity")
        else:
            print("   → Fairly equal distribution - dense attention")
        
        print("\n4. EFFECTIVE RANK (Lower = More Sparse)")
        avg_rank = np.mean(metrics['effective_rank'])
        norm_rank = np.mean(metrics['normalized_rank'])
        print(f"   Effective Rank: {avg_rank:.2f}")
        print(f"   Normalized Rank: {norm_rank:.4f}")
        
        if norm_rank < 0.3:
            print(f"   → Using only ~{norm_rank*100:.1f}% of available dimensions")
            print(f"   → Strong candidate for low-rank attention approximation")
        elif norm_rank < 0.5:
            print(f"   → Using ~{norm_rank*100:.1f}% of dimensions - moderate compression possible")
        
        print("\n5. HOYER SPARSITY (Higher = More Sparse, range [0,1])")
        print(f"   Hoyer: {np.mean(metrics['hoyer_sparsity']):.4f} ± {np.std(metrics['hoyer_sparsity']):.4f}")
        
        if np.mean(metrics['hoyer_sparsity']) > 0.5:
            print("   → Mathematically sparse - suitable for sparse attention mechanisms")
        elif np.mean(metrics['hoyer_sparsity']) > 0.3:
            print("   → Moderate sparsity - pruning could help")
        else:
            print("   → Low sparsity - dense attention preferred")
        
        print("\n6. ZERO/NEAR-ZERO RATIO (Higher = More Sparse)")
        print(f"   Near-zero weights (<1%): {np.mean(metrics['sparsity_ratio'])*100:.2f}%")
        
        if np.mean(metrics['sparsity_ratio']) > 0.7:
            print("   → >70% of weights are negligible - high compression potential!")
        
        # OVERALL RECOMMENDATION
        print("\n" + "="*70)
        print("OVERALL SPARSITY ASSESSMENT")
        print("="*70)
        
        # Score each metric (0-1, where 1 = most sparse)
        scores = {
            'entropy': 1 - np.mean(metrics['normalized_entropy']),
            'top_k': np.mean(metrics['top_k_mass'][10]),
            'gini': np.mean(metrics['gini']),
            'rank': 1 - norm_rank,
            'hoyer': np.mean(metrics['hoyer_sparsity']),
            'zeros': np.mean(metrics['sparsity_ratio'])
        }
        
        overall_score = np.mean(list(scores.values()))
        
        print(f"\nSparsity Score: {overall_score:.3f}/1.000")
        print(f"  (0.0 = completely dense, 1.0 = completely sparse)\n")
        
        if overall_score > 0.6:
            print("✓ HIGHLY SUITABLE for sparse attention!")
            print("  Recommendations:")
            print("  - Top-K attention with K=10-20")
            print("  - Sparse attention patterns (local + global)")
            print("  - Attention pruning/distillation")
            print(f"  - Expected speedup: {1/(1-np.mean(metrics['sparsity_ratio'])):.1f}x")
        elif overall_score > 0.4:
            print("✓ MODERATELY SUITABLE for sparse attention")
            print("  Recommendations:")
            print("  - Structured sparsity (block-sparse)")
            print("  - Low-rank approximations")
            print("  - Selective attention pruning")
        else:
            print("✗ NOT VERY SUITABLE for sparse attention")
            print("  Model uses distributed attention patterns")
            print("  Full attention likely necessary for performance")
        
        print("="*70)
        
        return overall_score
    
    def visualize_sparsity(self, metrics, output_dir):
        """Create sparsity visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Entropy distribution
        axes[0].hist(metrics['normalized_entropy'], bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
        axes[0].axvline(np.mean(metrics['normalized_entropy']), color='red', linestyle='--', label='Mean')
        axes[0].set_xlabel('Normalized Entropy')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Attention Entropy Distribution', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Top-K mass
        k_values = [1, 5, 10, 20, 50]
        k_means = [np.mean(metrics['top_k_mass'][k]) for k in k_values]
        axes[1].plot(k_values, k_means, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
        axes[1].set_xlabel('K (Number of Top Positions)')
        axes[1].set_ylabel('Fraction of Total Attention')
        axes[1].set_title('Top-K Attention Concentration', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # 3. Gini coefficient
        axes[2].hist(metrics['gini'], bins=30, alpha=0.7, color='#FFA07A', edgecolor='black')
        axes[2].axvline(np.mean(metrics['gini']), color='red', linestyle='--', label='Mean')
        axes[2].set_xlabel('Gini Coefficient')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Attention Inequality (Gini)', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Effective rank
        axes[3].hist(metrics['normalized_rank'], bins=30, alpha=0.7, color='#45B7D1', edgecolor='black')
        axes[3].axvline(np.mean(metrics['normalized_rank']), color='red', linestyle='--', label='Mean')
        axes[3].set_xlabel('Normalized Effective Rank')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Attention Dimensionality', fontweight='bold')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 5. Hoyer sparsity
        axes[4].hist(metrics['hoyer_sparsity'], bins=30, alpha=0.7, color='#95E1D3', edgecolor='black')
        axes[4].axvline(np.mean(metrics['hoyer_sparsity']), color='red', linestyle='--', label='Mean')
        axes[4].set_xlabel('Hoyer Sparsity')
        axes[4].set_ylabel('Frequency')
        axes[4].set_title('Mathematical Sparsity (Hoyer)', fontweight='bold')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # 6. Sparsity ratio
        axes[5].hist(np.array(metrics['sparsity_ratio'])*100, bins=30, alpha=0.7, color='#F38181', edgecolor='black')
        axes[5].axvline(np.mean(metrics['sparsity_ratio'])*100, color='red', linestyle='--', label='Mean')
        axes[5].set_xlabel('Percentage of Near-Zero Weights')
        axes[5].set_ylabel('Frequency')
        axes[5].set_title('Zero/Near-Zero Weight Ratio', fontweight='bold')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sparsity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to {output_dir}/sparsity_analysis.png")
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description='Analyze attention sparsity')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json')
    parser.add_argument('--ppg_file', type=str,
                        default='../../data/mesa_processed/mesa_ppg_with_labels.h5')
    parser.add_argument('--ecg_file', type=str,
                        default='../../data/mesa_processed/mesa_real_ecg.h5')
    parser.add_argument('--index_file', type=str,
                        default='../../data/mesa_processed/mesa_subject_index.h5')
    parser.add_argument('--subject_id', type=int, default=1)
    parser.add_argument('--n_windows', type=int, default=None)
    parser.add_argument('--output_dir', type=str,
                        default='../../outputs/sparsity_analysis')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ATTENTION SPARSITY ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    analyzer = AttentionSparsityAnalyzer(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Analyze
    metrics = analyzer.analyze_subject_sparsity(
        h5_ppg_path=args.ppg_file,
        h5_ecg_path=args.ecg_file,
        h5_index_path=args.index_file,
        subject_id=args.subject_id,
        n_windows=args.n_windows
    )
    
    # Report
    score = analyzer.print_sparsity_report(metrics)
    
    # Visualize
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    analyzer.visualize_sparsity(metrics, output_dir)
    
    # Save metrics
    import os
    
    def make_json_serializable(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.ndarray):
            return [float(x) for x in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(x) for x in obj]
        else:
            return obj
    
    metrics_serializable = make_json_serializable(metrics)
    
    with open(os.path.join(output_dir, 'sparsity_metrics.json'), 'w') as f:
        json.dump({
            'overall_score': float(score),
            'metrics': metrics_serializable
        }, f, indent=2)
    
    # Cleanup
    analyzer.cleanup()
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
