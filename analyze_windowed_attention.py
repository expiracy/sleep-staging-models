"""
Attention Analysis for Windowed Cross-Attention Model

This script extracts and visualizes attention patterns from the windowed model to understand:
1. How PPG and ECG modalities attend to each other
2. Attention patterns across different sleep stages
3. Temporal attention dynamics within windows
4. Modality weighting patterns
"""

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

from models.multimodal_model_crossattn_windowed import WindowAdaptiveSleepNet

# Sleep stage labels
SLEEP_STAGES = {
    0: 'Wake',
    1: 'N1/N2 (Light)',
    2: 'N3 (Deep)',
    3: 'REM'
}

STAGE_COLORS = {
    0: '#FF6B6B',  # Wake - Red
    1: '#4ECDC4',  # N1/N2 - Teal
    2: '#45B7D1',  # N3 - Blue
    3: '#FFA07A'   # REM - Orange
}


class WindowedAttentionAnalyzer:
    """Analyze attention patterns in windowed cross-attention model"""
    
    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        """
        Initialize attention analyzer
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config.json
            device: Device to run on
        """
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
        
        # Storage for attention weights
        self.attention_weights = {}
        self.modality_weights = {}
        
        # Register hooks to capture attention
        self._register_hooks()
    
    def _load_model(self):
        """Load the windowed model"""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get model parameters
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
        
        print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        print(f"Parameters: d_model={d_model}, n_heads={n_heads}, n_fusion_blocks={n_fusion_blocks}")
        
        return model
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        self.hooks = []
        
        # Hook into each fusion block's cross-attention layers
        for idx, fusion_block in enumerate(self.model.fusion_blocks):
            # PPG cross-attention
            def make_ppg_hook(block_idx):
                def hook(module, input, output):
                    # output is (attended_features, attention_weights)
                    if isinstance(output, tuple) and len(output) == 2:
                        self.attention_weights[f'block_{block_idx}_ppg'] = output[1].detach().cpu()
                return hook
            
            # ECG cross-attention  
            def make_ecg_hook(block_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) == 2:
                        self.attention_weights[f'block_{block_idx}_ecg'] = output[1].detach().cpu()
                return hook
            
            self.hooks.append(
                fusion_block.ppg_cross_attn.register_forward_hook(make_ppg_hook(idx))
            )
            self.hooks.append(
                fusion_block.ecg_cross_attn.register_forward_hook(make_ecg_hook(idx))
            )
        
        # Hook modality weighting
        def modality_hook(module, input, output):
            ppg_weight, ecg_weight = output
            self.modality_weights['ppg'] = ppg_weight.detach().cpu()
            self.modality_weights['ecg'] = ecg_weight.detach().cpu()
        
        self.hooks.append(
            self.model.modality_weighting.register_forward_hook(modality_hook)
        )
    
    def analyze_window(self, ppg_data, ecg_data, labels, window_idx=0):
        """
        Analyze attention for a single window
        
        Args:
            ppg_data: PPG signal (samples,)
            ecg_data: ECG signal (samples,)
            labels: Sleep stage labels (n_epochs,)
            window_idx: Window identifier
        
        Returns:
            Dictionary with analysis results
        """
        # Clear previous attention weights
        self.attention_weights = {}
        self.modality_weights = {}
        
        # Prepare input
        ppg_tensor = torch.FloatTensor(ppg_data).unsqueeze(0).unsqueeze(0).to(self.device)
        ecg_tensor = torch.FloatTensor(ecg_data).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(ppg_tensor, ecg_tensor)
        
        # Get predictions
        predictions = outputs.squeeze(0).argmax(dim=0).cpu().numpy()
        probabilities = outputs.squeeze(0).cpu().numpy()
        
        return {
            'window_idx': window_idx,
            'predictions': predictions,
            'probabilities': probabilities,
            'labels': labels,
            'attention_weights': self.attention_weights.copy(),
            'modality_weights': self.modality_weights.copy(),
            'n_epochs': len(labels)
        }
    
    def analyze_subject(self, h5_ppg_path, h5_ecg_path, h5_index_path, subject_id, n_windows=None):
        """
        Analyze attention patterns for a subject
        
        Args:
            h5_ppg_path: Path to PPG H5 file
            h5_ecg_path: Path to ECG H5 file
            h5_index_path: Path to index H5 file
            subject_id: Subject ID to analyze
            n_windows: Number of windows to analyze (None = all)
        
        Returns:
            List of window analysis results
        """
        print(f"\nAnalyzing subject {subject_id}")
        
        # Load subject data
        with h5py.File(h5_index_path, 'r') as f:
            subject_key = f'{subject_id:04d}'
            if f'subjects/{subject_key}' not in f:
                raise ValueError(f"Subject {subject_key} not found")
            
            window_indices = f[f'subjects/{subject_key}/window_indices'][:]
            start_idx = window_indices[0]
            n_epochs = len(window_indices)
        
        # Load data
        with h5py.File(h5_ppg_path, 'r') as f:
            ppg_epochs = f['ppg'][start_idx:start_idx + n_epochs]
            labels = f['labels'][start_idx:start_idx + n_epochs]
        
        with h5py.File(h5_ecg_path, 'r') as f:
            ecg_epochs = f['ecg'][start_idx:start_idx + n_epochs]
        
        # Get window configuration
        window_epochs = self.config['windowing']['window_epochs']
        overlap_percent = self.config['windowing']['overlap_percent']
        
        samples_per_epoch = 1024
        chunk_size = window_epochs * samples_per_epoch
        overlap_epochs = int(window_epochs * overlap_percent / 100)
        stride = (window_epochs - overlap_epochs) * samples_per_epoch
        
        print(f"Window config: {window_epochs} epochs, {overlap_percent}% overlap")
        print(f"Total epochs: {n_epochs}")
        
        # Analyze windows
        results = []
        ppg_continuous = ppg_epochs.reshape(-1)
        ecg_continuous = ecg_epochs.reshape(-1)
        
        window_count = 0
        for start_idx in tqdm(range(0, len(ppg_continuous) - chunk_size + 1, stride), desc="Analyzing windows"):
            ppg_window = ppg_continuous[start_idx:start_idx + chunk_size]
            ecg_window = ecg_continuous[start_idx:start_idx + chunk_size]
            
            start_epoch = start_idx // samples_per_epoch
            end_epoch = start_epoch + window_epochs
            labels_window = labels[start_epoch:end_epoch]
            
            result = self.analyze_window(ppg_window, ecg_window, labels_window, window_count)
            results.append(result)
            
            window_count += 1
            if n_windows and window_count >= n_windows:
                break
        
        print(f"Analyzed {len(results)} windows")
        
        return results
    
    def visualize_attention_patterns(self, results, output_dir):
        """
        Create comprehensive attention visualizations
        
        Args:
            results: List of window analysis results
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nCreating visualizations in {output_dir}")
        
        # 1. Average attention by sleep stage
        self._plot_attention_by_sleep_stage(results, output_dir)
        
        # 2. Attention heatmaps for each fusion block
        self._plot_attention_heatmaps(results, output_dir)
        
        # 3. Modality weights distribution
        self._plot_modality_weights(results, output_dir)
        
        # 4. Temporal attention evolution
        self._plot_temporal_attention(results, output_dir)
        
        # 5. Per-window attention summary
        self._plot_window_attention_summary(results, output_dir)
        
        print("Visualizations complete!")
    
    def _plot_attention_by_sleep_stage(self, results, output_dir):
        """Plot average attention patterns by sleep stage"""
        print("Plotting attention by sleep stage...")
        
        # Aggregate attention by sleep stage
        stage_attentions = {stage: [] for stage in range(4)}
        
        for result in results:
            labels = result['labels']
            
            # Get attention from first fusion block (PPG attending to ECG)
            if 'block_0_ppg' in result['attention_weights']:
                attn = result['attention_weights']['block_0_ppg']
                # attn shape: (batch=1, n_heads, seq_len, seq_len)
                attn_avg = attn[0].mean(dim=0).numpy()  # Average over heads
                
                # Group by sleep stage
                for epoch_idx, label in enumerate(labels):
                    if label != -1 and epoch_idx < attn_avg.shape[0]:
                        stage_attentions[label].append(attn_avg[epoch_idx])
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for stage, ax in zip(range(4), axes):
            if stage_attentions[stage]:
                avg_attn = np.mean(stage_attentions[stage], axis=0)
                
                im = ax.imshow(avg_attn.reshape(1, -1), aspect='auto', cmap='viridis')
                ax.set_title(f'{SLEEP_STAGES[stage]} - Avg Attention Pattern', fontsize=12, fontweight='bold')
                ax.set_xlabel('Attended Position')
                ax.set_ylabel('Query')
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, label='Attention Weight')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{SLEEP_STAGES[stage]} - No data')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_by_sleep_stage.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_heatmaps(self, results, output_dir):
        """Plot attention heatmaps for each fusion block"""
        print("Plotting attention heatmaps...")
        
        # Get number of fusion blocks
        n_blocks = len([k for k in results[0]['attention_weights'].keys() if 'ppg' in k])
        
        for block_idx in range(n_blocks):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Collect attention weights across all windows
            ppg_attns = []
            ecg_attns = []
            
            for result in results:
                ppg_key = f'block_{block_idx}_ppg'
                ecg_key = f'block_{block_idx}_ecg'
                
                if ppg_key in result['attention_weights']:
                    attn = result['attention_weights'][ppg_key]
                    ppg_attns.append(attn[0].mean(dim=0).numpy())  # Avg over heads
                
                if ecg_key in result['attention_weights']:
                    attn = result['attention_weights'][ecg_key]
                    ecg_attns.append(attn[0].mean(dim=0).numpy())
            
            # Average and plot
            if ppg_attns:
                avg_ppg = np.mean(ppg_attns, axis=0)
                im1 = ax1.imshow(avg_ppg, cmap='viridis', aspect='auto')
                ax1.set_title(f'Block {block_idx} - PPG→ECG Attention', fontsize=12, fontweight='bold')
                ax1.set_xlabel('ECG Position (Key)')
                ax1.set_ylabel('PPG Position (Query)')
                plt.colorbar(im1, ax=ax1, label='Attention Weight')
            
            if ecg_attns:
                avg_ecg = np.mean(ecg_attns, axis=0)
                im2 = ax2.imshow(avg_ecg, cmap='plasma', aspect='auto')
                ax2.set_title(f'Block {block_idx} - ECG→PPG Attention', fontsize=12, fontweight='bold')
                ax2.set_xlabel('PPG Position (Key)')
                ax2.set_ylabel('ECG Position (Query)')
                plt.colorbar(im2, ax=ax2, label='Attention Weight')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'attention_heatmap_block{block_idx}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_modality_weights(self, results, output_dir):
        """Plot distribution of modality weights"""
        print("Plotting modality weights...")
        
        ppg_weights = []
        ecg_weights = []
        
        for result in results:
            if 'ppg' in result['modality_weights']:
                ppg_weights.append(result['modality_weights']['ppg'].item())
                ecg_weights.append(result['modality_weights']['ecg'].item())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(ppg_weights, bins=30, alpha=0.6, label='PPG', color='#4ECDC4')
        axes[0].hist(ecg_weights, bins=30, alpha=0.6, label='ECG', color='#FF6B6B')
        axes[0].set_xlabel('Weight Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Modality Weight Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Time series
        axes[1].plot(ppg_weights, label='PPG', color='#4ECDC4', alpha=0.7)
        axes[1].plot(ecg_weights, label='ECG', color='#FF6B6B', alpha=0.7)
        axes[1].set_xlabel('Window Index')
        axes[1].set_ylabel('Weight Value')
        axes[1].set_title('Modality Weights Across Windows', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'modality_weights.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print(f"\nModality Weight Statistics:")
        print(f"  PPG: {np.mean(ppg_weights):.4f} ± {np.std(ppg_weights):.4f}")
        print(f"  ECG: {np.mean(ecg_weights):.4f} ± {np.std(ecg_weights):.4f}")
    
    def _plot_temporal_attention(self, results, output_dir):
        """Plot temporal evolution of attention patterns"""
        print("Plotting temporal attention evolution...")
        
        # Sample a few windows for detailed analysis
        sample_indices = np.linspace(0, len(results)-1, min(4, len(results)), dtype=int)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, window_idx in enumerate(sample_indices):
            result = results[window_idx]
            
            if 'block_0_ppg' in result['attention_weights']:
                attn = result['attention_weights']['block_0_ppg']
                attn_avg = attn[0].mean(dim=0).numpy()  # Avg over heads
                
                im = axes[idx].imshow(attn_avg, cmap='viridis', aspect='auto')
                axes[idx].set_title(f'Window {window_idx} Attention Pattern', fontsize=11, fontweight='bold')
                axes[idx].set_xlabel('Position (Key)')
                axes[idx].set_ylabel('Position (Query)')
                plt.colorbar(im, ax=axes[idx], label='Attention')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_attention_samples.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_window_attention_summary(self, results, output_dir):
        """Plot per-window attention statistics summary"""
        print("Plotting window attention summary...")
        
        window_stats = []
        
        for result in results:
            if 'block_0_ppg' in result['attention_weights']:
                attn = result['attention_weights']['block_0_ppg']
                attn_numpy = attn[0].mean(dim=0).numpy()  # Avg over heads
                
                window_stats.append({
                    'max': attn_numpy.max(),
                    'mean': attn_numpy.mean(),
                    'std': attn_numpy.std(),
                    'entropy': -np.sum(attn_numpy * np.log(attn_numpy + 1e-10))
                })
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['max', 'mean', 'std', 'entropy']
        titles = ['Max Attention', 'Mean Attention', 'Std Attention', 'Attention Entropy']
        
        for ax, metric, title in zip(axes.flatten(), metrics, titles):
            values = [s[metric] for s in window_stats]
            ax.plot(values, linewidth=1.5, color='#4ECDC4')
            ax.set_xlabel('Window Index')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Across Windows', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'window_attention_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results, output_dir):
        """Save analysis results to JSON"""
        # Prepare serializable results
        save_data = {
            'n_windows': len(results),
            'config': self.config,
            'windows': []
        }
        
        for result in results:
            window_data = {
                'window_idx': result['window_idx'],
                'n_epochs': result['n_epochs'],
                'predictions': result['predictions'].tolist(),
                'labels': result['labels'].tolist(),
                'modality_weights': {
                    'ppg': float(result['modality_weights']['ppg'].item()) if 'ppg' in result['modality_weights'] else None,
                    'ecg': float(result['modality_weights']['ecg'].item()) if 'ecg' in result['modality_weights'] else None
                }
            }
            save_data['windows'].append(window_data)
        
        output_file = os.path.join(output_dir, 'attention_analysis_results.json')
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description='Analyze windowed model attention patterns')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to windowed model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (auto-detects if not provided)')
    parser.add_argument('--ppg_file', type=str,
                        default='../../data/mesa_processed/mesa_ppg_with_labels.h5',
                        help='Path to PPG H5 file')
    parser.add_argument('--ecg_file', type=str,
                        default='../../data/mesa_processed/mesa_real_ecg.h5',
                        help='Path to ECG H5 file')
    parser.add_argument('--index_file', type=str,
                        default='../../data/mesa_processed/mesa_subject_index.h5',
                        help='Path to index H5 file')
    parser.add_argument('--subject_id', type=int, default=1,
                        help='Subject ID to analyze')
    parser.add_argument('--n_windows', type=int, default=None,
                        help='Number of windows to analyze (None = all)')
    parser.add_argument('--output_dir', type=str,
                        default='../../outputs/windowed_attention_analysis',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Windowed Model Attention Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = WindowedAttentionAnalyzer(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Analyze subject
    results = analyzer.analyze_subject(
        h5_ppg_path=args.ppg_file,
        h5_ecg_path=args.ecg_file,
        h5_index_path=args.index_file,
        subject_id=args.subject_id,
        n_windows=args.n_windows
    )
    
    # Create visualizations
    analyzer.visualize_attention_patterns(results, output_dir)
    
    # Save results
    analyzer.save_results(results, output_dir)
    
    # Cleanup
    analyzer.cleanup()
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
