"""
inference.py
Load trained sleep staging models and perform inference on PPG data

This script demonstrates how to:
1. Load a trained model checkpoint
2. Prepare PPG data for inference
3. Generate sleep stage predictions
4. Visualize and save results
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
import time
import psutil
import tracemalloc

# Set number of threads for PyTorch operations
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Import model architectures
from models.multimodal_sleep_model import SleepPPGNet
from models.ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention
from models.multimodal_model_crossattn import ImprovedMultiModalSleepNet
from models.multimodal_model_crossattn_windowed import WindowAdaptiveSleepNet

# Sleep stage labels
SLEEP_STAGES = {
    0: 'Wake',
    1: 'N1/N2',
    2: 'N3',
    3: 'REM'
}

SLEEP_STAGE_COLORS = {
    0: '#FF6B6B',  # Wake - Red
    1: '#4ECDC4',  # N1/N2 - Teal
    2: '#45B7D1',  # N3 - Blue
    3: '#FFA07A'   # REM - Orange
}


class SleepStageInference:
    """Class for loading models and performing sleep stage inference"""
    
    def __init__(self, checkpoint_path, model_type=None, device='cuda', monitor_resources=True, config_path=None):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            model_type: Type of model ('ppg_only', 'ppg_unfiltered', 'crossattn_ecg', 'windowed_crossattn')
                       If None, will try to auto-detect from config.json
            device: Device to run inference on ('cuda' or 'cpu')
            monitor_resources: If True, monitor memory and execution time
            config_path: Path to config.json (if None, will look for it next to checkpoint)
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.monitor_resources = monitor_resources
        
        # Load config if provided or auto-detect
        self.config = self._load_config(config_path)
        
        # Auto-detect model type from config if not provided
        if model_type is None and self.config and 'model_type' in self.config:
            self.model_type = self.config['model_type']
            print(f"Auto-detected model type from config: {self.model_type}")
        elif model_type is None:
            raise ValueError("model_type not provided and could not be auto-detected from config.json")
        else:
            self.model_type = model_type
        
        print(f"Initializing inference on {self.device}")
        
        # Initialize monitoring
        if self.monitor_resources:
            self.process = psutil.Process(os.getpid())
            self.metrics = {
                'load_time': 0,
                'preprocess_time': 0,
                'inference_time': 0,
                'postprocess_time': 0,
                'total_time': 0,
                'peak_memory_mb': 0,
                'initial_memory_mb': 0,
                'final_memory_mb': 0
            }
            tracemalloc.start()
            self.metrics['initial_memory_mb'] = self.process.memory_info().rss / 1024 / 1024
        
        # Load model
        start_time = time.time()
        self.model = self._load_model()
        self.model.eval()
        if self.monitor_resources:
            self.metrics['load_time'] = time.time() - start_time
        
        # Load checkpoint metadata
        self.checkpoint_info = self._load_checkpoint_info()
    
    def _load_config(self, config_path):
        """Load config.json from checkpoint directory or specified path"""
        if config_path is None:
            # Try to find config.json in checkpoint directory
            checkpoint_dir = Path(self.checkpoint_path).parent
            config_path = checkpoint_dir / 'config.json'
        else:
            config_path = Path(config_path)
        
        if config_path.exists():
            print(f"Loading config from {config_path}")
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: config.json not found at {config_path}")
            return None
        
    def _load_model(self):
        """Load the appropriate model architecture and weights"""
        print(f"\nLoading {self.model_type} model from {self.checkpoint_path}")
        
        # Load checkpoint (weights_only=False for compatibility with older checkpoints)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get model parameters from config if available
        if self.config and 'model' in self.config:
            model_params = self.config['model']
            d_model = model_params.get('d_model', 256)
            n_heads = model_params.get('n_heads', 8)
            n_fusion_blocks = model_params.get('n_fusion_blocks', 3)
        else:
            # Default parameters
            d_model = 256
            n_heads = 8
            n_fusion_blocks = 3
        
        # Create model based on type
        if self.model_type == 'ppg_only':
            model = SleepPPGNet()
        elif self.model_type == 'ppg_unfiltered':
            model = PPGUnfilteredCrossAttention()
        elif self.model_type == 'crossattn_ecg':
            model = ImprovedMultiModalSleepNet(
                n_classes=4,
                d_model=d_model,
                n_heads=n_heads,
                n_fusion_blocks=n_fusion_blocks
            )
        elif self.model_type == 'windowed_crossattn':
            model = WindowAdaptiveSleepNet(
                n_classes=4,
                d_model=d_model,
                n_heads=n_heads,
                n_fusion_blocks=n_fusion_blocks
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        
        # Print model info
        if 'epoch' in checkpoint:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'best_val_kappa' in checkpoint:
            print(f"Best validation kappa: {checkpoint['best_val_kappa']:.4f}")
        if 'best_val_acc' in checkpoint:
            print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        return model
    
    def _load_checkpoint_info(self):
        """Load checkpoint metadata"""
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        return {
            'epoch': checkpoint.get('epoch', 'Unknown'),
            'val_kappa': checkpoint.get('best_val_kappa', 'Unknown'),
            'val_acc': checkpoint.get('best_val_acc', 'Unknown')
        }
    
    def _preprocess_ppg(self, ppg_data):
        """
        Preprocess PPG data for model input
        
        Args:
            ppg_data: Raw PPG signal as numpy array (continuous signal)
        
        Returns:
            ppg_tensor: Preprocessed tensor ready for model input
        """
        # Ensure ppg_data is a numpy array
        if not isinstance(ppg_data, np.ndarray):
            ppg_data = np.array(ppg_data)
        
        # Reshape to windows if needed
        # Expected shape: (batch, channels, samples) or (batch, samples)
        # For PPG models, we typically have (1, 1, N) where N is the signal length
        
        # Convert to tensor
        ppg_tensor = torch.from_numpy(ppg_data).float()
        
        # Add batch and channel dimensions if not present
        if ppg_tensor.dim() == 1:
            ppg_tensor = ppg_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        elif ppg_tensor.dim() == 2:
            ppg_tensor = ppg_tensor.unsqueeze(1)  # (B, 1, N)
        
        # Move to device
        ppg_tensor = ppg_tensor.to(self.device)
        
        return ppg_tensor

    def predict(self, ppg_data, return_probabilities=False, streaming=False, window_size_epochs=None, overlap_epochs=None, batch_size=1):
        """
        Perform sleep stage prediction
        
        Args:
            ppg_data: Raw PPG signal (numpy array)
            return_probabilities: If True, return class probabilities
            streaming: If True and model is windowed, process in chunks to save memory
            window_size_epochs: Number of epochs per chunk (default: 30 for 15-min windows)
            overlap_epochs: Number of overlapping epochs between chunks (default: 3 for 10% overlap)
            batch_size: Number of chunks to process in parallel (default: 1, higher = faster but more memory)
        
        Returns:
            predictions: Sleep stage predictions for each 30-second epoch
            probabilities (optional): Class probabilities for each epoch
        """
        # Use streaming mode for windowed models if enabled
        if streaming and self.model_type == 'windowed_crossattn':
            return self._predict_streaming(ppg_data, return_probabilities, window_size_epochs, overlap_epochs, batch_size)
        
        # Standard inference (full sequence)
        if self.monitor_resources:
            start_time = time.time()
            
        ppg_tensor = self._preprocess_ppg(ppg_data)
        
        if self.monitor_resources:
            self.metrics['preprocess_time'] += time.time() - start_time
            start_time = time.time()
        
        # Run inference
        with torch.no_grad():
            if self.model_type == 'ppg_only':
                outputs = self.model(ppg_tensor)
            elif self.model_type == 'ppg_unfiltered':
                outputs = self.model(ppg_tensor)
            elif self.model_type in ['crossattn_ecg', 'windowed_crossattn']:
                outputs = self.model(ppg_tensor, ppg_tensor)
        
        if self.monitor_resources:
            self.metrics['inference_time'] += time.time() - start_time
            start_time = time.time()
        
        # outputs shape: (batch, n_classes, n_epochs)
        # Get predictions
        probabilities = outputs.squeeze(0).cpu().numpy()  # (n_classes, n_epochs)
        predictions = np.argmax(probabilities, axis=0)  # (n_epochs,)
        
        if self.monitor_resources:
            self.metrics['postprocess_time'] += time.time() - start_time
            # Update memory metrics
            current_memory = self.process.memory_info().rss / 1024 / 1024
            self.metrics['peak_memory_mb'] = max(self.metrics['peak_memory_mb'], current_memory)
        
        if return_probabilities:
            return predictions, probabilities.T  # (n_epochs, n_classes)
        else:
            return predictions
    
    def _predict_streaming(self, ppg_data, return_probabilities=False, window_size_epochs=None, overlap_epochs=None, batch_size=1):
        """
        Streaming inference for windowed models - processes chunks in batches
        
        Args:
            ppg_data: Raw PPG signal (numpy array)
            return_probabilities: If True, return class probabilities
            window_size_epochs: Number of epochs per chunk (default: 30 for 15-min)
            overlap_epochs: Number of overlapping epochs (default: 3 for 10%)
            batch_size: Number of chunks to process in parallel (higher = faster, more memory)
        
        Returns:
            predictions: Sleep stage predictions for each 30-second epoch
            probabilities (optional): Class probabilities for each epoch
        """
        samples_per_epoch = 1024
        total_samples = len(ppg_data)
        total_epochs = total_samples // samples_per_epoch
        
        # Default window parameters (15-minute windows with 10% overlap)
        if window_size_epochs is None:
            window_size_epochs = 30
        if overlap_epochs is None:
            overlap_epochs = max(1, int(window_size_epochs * 0.1))
        
        stride_epochs = window_size_epochs - overlap_epochs
        window_size_samples = window_size_epochs * samples_per_epoch
        stride_samples = stride_epochs * samples_per_epoch
        
        # Calculate chunks
        chunk_starts = list(range(0, total_samples - window_size_samples + 1, stride_samples))
        num_chunks = len(chunk_starts)
        
        print(f"\nStreaming Inference Configuration:")
        print(f"  Window size: {window_size_epochs} epochs ({window_size_epochs * 0.5:.1f} min)")
        print(f"  Overlap: {overlap_epochs} epochs ({overlap_epochs * 0.5:.1f} min)")
        print(f"  Stride: {stride_epochs} epochs")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Total chunks: {num_chunks}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches: {(num_chunks + batch_size - 1) // batch_size}")
        
        # Initialize output arrays
        all_predictions = []
        all_probabilities = [] if return_probabilities else None
        
        # Process in batches
        for batch_idx in range(0, num_chunks, batch_size):
            batch_end = min(batch_idx + batch_size, num_chunks)
            current_batch_size = batch_end - batch_idx
            
            # Prepare batch of chunks
            batch_chunks = []
            for i in range(batch_idx, batch_end):
                start_sample = chunk_starts[i]
                end_sample = start_sample + window_size_samples
                chunk_data = ppg_data[start_sample:end_sample]
                batch_chunks.append(chunk_data)
            
            # Stack into batch tensor
            if self.monitor_resources:
                start_time = time.time()
            
            batch_array = np.stack(batch_chunks, axis=0)  # (batch_size, window_size_samples)
            ppg_batch = self._preprocess_ppg(batch_array)  # (batch_size, 1, window_size_samples)
            
            if self.monitor_resources:
                self.metrics['preprocess_time'] += time.time() - start_time
                start_time = time.time()
            
            # Run inference on batch
            with torch.no_grad():
                outputs = self.model(ppg_batch, ppg_batch)  # (batch_size, n_classes, window_size_epochs)
            
            if self.monitor_resources:
                self.metrics['inference_time'] += time.time() - start_time
                start_time = time.time()
            
            # Extract predictions for each chunk in batch
            for chunk_offset in range(current_batch_size):
                chunk_probs = outputs[chunk_offset].cpu().numpy()  # (n_classes, window_size_epochs)
                chunk_preds = np.argmax(chunk_probs, axis=0)       # (window_size_epochs,)
                
                global_chunk_idx = batch_idx + chunk_offset
                
                # For overlapping regions, skip overlap except for first chunk
                if global_chunk_idx == 0:
                    # First chunk - use all predictions
                    all_predictions.append(chunk_preds)
                    if return_probabilities:
                        all_probabilities.append(chunk_probs.T)  # (window_size_epochs, n_classes)
                else:
                    # Subsequent chunks - skip overlap region
                    all_predictions.append(chunk_preds[overlap_epochs:])
                    if return_probabilities:
                        all_probabilities.append(chunk_probs.T[overlap_epochs:])
            
            if self.monitor_resources:
                self.metrics['postprocess_time'] += time.time() - start_time
                current_memory = self.process.memory_info().rss / 1024 / 1024
                self.metrics['peak_memory_mb'] = max(self.metrics['peak_memory_mb'], current_memory)
            
            # Clear GPU cache after each batch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            start_epoch = chunk_starts[batch_idx] // samples_per_epoch
            end_epoch = (chunk_starts[batch_end - 1] + window_size_samples) // samples_per_epoch - 1
            print(f"  Processed batch {(batch_idx // batch_size) + 1}: chunks {batch_idx + 1}-{batch_end}, epochs {start_epoch}-{end_epoch}")
        
        # Concatenate all predictions
        predictions = np.concatenate(all_predictions)
        if return_probabilities:
            probabilities = np.concatenate(all_probabilities, axis=0)
        
        # Handle any remaining epochs not covered by windows
        covered_epochs = len(predictions)
        if covered_epochs < total_epochs:
            # Process final partial window if needed
            remaining_start = covered_epochs * samples_per_epoch
            remaining_data = ppg_data[remaining_start:]
            
            if len(remaining_data) >= samples_per_epoch:  # At least 1 epoch remains
                # Pad to window size if needed
                remaining_samples = len(remaining_data)
                if remaining_samples < window_size_samples:
                    # Pad with zeros
                    padding = np.zeros(window_size_samples - remaining_samples)
                    remaining_data = np.concatenate([remaining_data, padding])
                
                # Process remaining chunk
                ppg_tensor = self._preprocess_ppg(remaining_data[:window_size_samples])
                
                with torch.no_grad():
                    outputs = self.model(ppg_tensor, ppg_tensor)
                
                chunk_probs = outputs.squeeze(0).cpu().numpy()
                chunk_preds = np.argmax(chunk_probs, axis=0)
                
                # Only take the number of epochs we actually need
                remaining_epochs = total_epochs - covered_epochs
                predictions = np.concatenate([predictions, chunk_preds[:remaining_epochs]])
                
                if return_probabilities:
                    probabilities = np.concatenate([probabilities, chunk_probs.T[:remaining_epochs]], axis=0)
                
                print(f"  Processed final partial chunk: epochs {covered_epochs}-{total_epochs - 1}")
        
        # Ensure we have exactly total_epochs predictions
        predictions = predictions[:total_epochs]
        
        if return_probabilities:
            probabilities = probabilities[:total_epochs]
            return predictions, probabilities
        else:
            return predictions
    
    def predict_from_h5_subject(self, h5_file_path, h5_index_file_path, subject_id, return_probabilities=False, streaming=False, window_size_epochs=None, overlap_epochs=None, batch_size=1):
        """
        Load and predict for a subject from H5 file
        
        Args:
            h5_file_path: Path to H5 file with PPG data
            h5_index_file_path: Path to H5 index file
            subject_id: Subject ID to load (e.g. 1)
            return_probabilities: If True, return class probabilities
            streaming: If True, use streaming inference (memory efficient for windowed models)
            window_size_epochs: Window size for streaming (default: 30 epochs)
            overlap_epochs: Overlap for streaming (default: 3 epochs)
            batch_size: Number of chunks to process in parallel (default: 1, higher = faster)
        
        Returns:
            predictions: Sleep stage predictions
            true_labels (if available): Ground truth labels
            probabilities (optional): Class probabilities
        """
        if self.monitor_resources:
            total_start_time = time.time()
        
        print(f"\nLoading subject {subject_id} from {h5_file_path}")
        
        # Load subject indices
        with h5py.File(h5_index_file_path, 'r') as f:
            # Convert subject_id to the format used in the HDF5 file
            subject_key = f'{subject_id:04d}'
            
            # Check if subject exists
            if f'subjects/{subject_key}' not in f:
                # List available subjects
                available_subjects = list(f['subjects'].keys())[:10]
                raise ValueError(f"Subject {subject_key} not found. Available subjects (first 10): {available_subjects}")
            
            # Get window indices for this subject
            window_indices = f[f'subjects/{subject_key}/window_indices'][:]
            start_idx = window_indices[0]
            windows_per_subject = len(window_indices)
        
        # Load PPG data and labels
        with h5py.File(h5_file_path, 'r') as f:
            ppg_windows = f['ppg'][start_idx:start_idx + windows_per_subject]
            labels = f['labels'][start_idx:start_idx + windows_per_subject]
        
        # Reshape to continuous signal
        ppg_continuous = ppg_windows.reshape(-1)
        
        print(f"Loaded PPG data: {ppg_continuous.shape}")
        print(f"Loaded labels: {labels.shape}")
        
        # Predict
        if return_probabilities:
            predictions, probabilities = self.predict(ppg_continuous, return_probabilities=True, 
                                                     streaming=streaming, 
                                                     window_size_epochs=window_size_epochs,
                                                     overlap_epochs=overlap_epochs,
                                                     batch_size=batch_size)
            result = (predictions, labels, probabilities)
        else:
            predictions = self.predict(ppg_continuous, return_probabilities=False,
                                      streaming=streaming,
                                      window_size_epochs=window_size_epochs,
                                      overlap_epochs=overlap_epochs,
                                      batch_size=batch_size)
            result = (predictions, labels)
        
        if self.monitor_resources:
            self.metrics['total_time'] = time.time() - total_start_time
            self.metrics['final_memory_mb'] = self.process.memory_info().rss / 1024 / 1024
            self.print_resource_metrics()
        
        return result
    
    def visualize_predictions(self, predictions, true_labels=None, save_path=None):
        """
        Visualize sleep stage predictions as a hypnogram
        
        Args:
            predictions: Predicted sleep stages
            true_labels: Ground truth labels (optional)
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2 if true_labels is not None else 1, 1, 
                                 figsize=(15, 4 if true_labels is None else 8))
        
        if true_labels is None:
            axes = [axes]
        
        # Time axis (30-second epochs)
        time_hours = np.arange(len(predictions)) * 30 / 3600
        
        # Plot predictions
        for i in range(len(predictions)):
            color = SLEEP_STAGE_COLORS[predictions[i]]
            axes[0].axvspan(time_hours[i], time_hours[i] + 30/3600, 
                          facecolor=color, alpha=0.7)
        
        axes[0].set_ylim(-0.5, 3.5)
        axes[0].set_yticks([0, 1, 2, 3])
        axes[0].set_yticklabels(['Wake', 'N1/N2', 'N3', 'REM'])
        axes[0].set_xlabel('Time (hours)')
        axes[0].set_ylabel('Sleep Stage')
        axes[0].set_title('Predicted Sleep Stages (Hypnogram)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot ground truth if available
        if true_labels is not None:
            # Filter out invalid labels (-1)
            valid_mask = true_labels != -1
            valid_time = time_hours[valid_mask]
            valid_labels = true_labels[valid_mask]
            
            for i in range(len(valid_labels)):
                if i < len(valid_time) - 1:
                    color = SLEEP_STAGE_COLORS[valid_labels[i]]
                    axes[1].axvspan(valid_time[i], valid_time[i] + 30/3600,
                                  facecolor=color, alpha=0.7)
            
            axes[1].set_ylim(-0.5, 3.5)
            axes[1].set_yticks([0, 1, 2, 3])
            axes[1].set_yticklabels(['Wake', 'N1/N2', 'N3', 'REM'])
            axes[1].set_xlabel('Time (hours)')
            axes[1].set_ylabel('Sleep Stage')
            axes[1].set_title('Ground Truth Sleep Stages')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def compute_metrics(self, predictions, true_labels):
        """
        Compute evaluation metrics
        
        Args:
            predictions: Predicted sleep stages
            true_labels: Ground truth labels
        
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix
        
        # Filter out invalid labels
        valid_mask = true_labels != -1
        predictions = predictions[valid_mask]
        true_labels = true_labels[valid_mask]
        
        # Compute metrics
        kappa = cohen_kappa_score(true_labels, predictions)
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        
        # Per-class metrics
        f1_per_class = f1_score(true_labels, predictions, average=None)
        
        metrics = {
            'kappa': kappa,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': {
                SLEEP_STAGES[i]: f1_per_class[i] for i in range(len(f1_per_class))
            }
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print metrics in a nice format"""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Cohen's Kappa:      {metrics['kappa']:.4f}")
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")
        print(f"F1-Score (Weighted):{metrics['f1_weighted']:.4f}")
        print("\nPer-Class F1-Scores:")
        for stage, f1 in metrics['f1_per_class'].items():
            print(f"  {stage:8s}: {f1:.4f}")
        print("="*50)
    
    def print_resource_metrics(self):
        """Print resource usage metrics"""
        if not self.monitor_resources:
            print("Resource monitoring is disabled")
            return
        
        print("\n" + "="*50)
        print("RESOURCE USAGE METRICS")
        print("="*50)
        print(f"Device:              {self.device}")
        print(f"PyTorch Threads:     {torch.get_num_threads()}")
        print(f"Interop Threads:     {torch.get_num_interop_threads()}")
        print(f"\nTiming:")
        print(f"  Model Load Time:     {self.metrics['load_time']:.3f} seconds")
        print(f"  Preprocessing Time:  {self.metrics['preprocess_time']:.3f} seconds")
        print(f"  Inference Time:      {self.metrics['inference_time']:.3f} seconds")
        print(f"  Postprocessing Time: {self.metrics['postprocess_time']:.3f} seconds")
        print(f"  Total Time:          {self.metrics['total_time']:.3f} seconds")
        print(f"\nMemory Usage:")
        print(f"  Initial:  {self.metrics['initial_memory_mb']:.2f} MB")
        print(f"  Peak:     {self.metrics['peak_memory_mb']:.2f} MB")
        print(f"  Final:    {self.metrics['final_memory_mb']:.2f} MB")
        print(f"  Increase: {self.metrics['final_memory_mb'] - self.metrics['initial_memory_mb']:.2f} MB")
        
        # Get tracemalloc stats
        current, peak = tracemalloc.get_traced_memory()
        print(f"\nPython Memory (tracemalloc):")
        print(f"  Current:  {current / 1024 / 1024:.2f} MB")
        print(f"  Peak:     {peak / 1024 / 1024:.2f} MB")
        print("="*50)


def main():
    """Main inference script"""
    parser = argparse.ArgumentParser(description='Sleep Stage Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['ppg_only', 'ppg_unfiltered', 'crossattn_ecg', 'windowed_crossattn'],
                        help='Type of model (auto-detects from config.json if not provided)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (auto-detects from checkpoint dir if not provided)')
    parser.add_argument('--data_file', type=str,
                       default='../../data/mesa_processed/mesa_ppg_with_labels.h5',
                        help='Path to H5 data file')
    parser.add_argument('--data_index_file', type=str,
                        default='../../data/mesa_processed/mesa_subject_index.h5',
                        help='Path to H5 data index file')
    parser.add_argument('--subject_id', type=int, default=1,
                        help='Subject ID to test')
    parser.add_argument('--output_dir', type=str, default='../../outputs/inference_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming inference (memory efficient for windowed models)')
    parser.add_argument('--window_size_epochs', type=int, default=None,
                        help='Window size in epochs for streaming (default: 30)')
    parser.add_argument('--overlap_epochs', type=int, default=None,
                        help='Overlap in epochs for streaming (default: 3)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for streaming (higher = faster but more memory, default: 1)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference engine
    inference = SleepStageInference(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        monitor_resources=True,
        config_path=args.config
    )
    
    # Run prediction
    print(f"\nRunning inference on subject {args.subject_id}...")
    if args.streaming:
        print("Using STREAMING mode - processing chunks sequentially for lower memory usage")
    
    predictions, true_labels, probabilities = inference.predict_from_h5_subject(
        args.data_file,
        args.data_index_file,
        args.subject_id,
        return_probabilities=True,
        streaming=args.streaming,
        window_size_epochs=args.window_size_epochs,
        overlap_epochs=args.overlap_epochs,
        batch_size=args.batch_size
    )
    
    # Compute metrics
    metrics = inference.compute_metrics(predictions, true_labels)
    inference.print_metrics(metrics)
    
    # Visualize results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.output_dir, 
                            f'hypnogram_subject_{args.subject_id}_{timestamp}.png')
    inference.visualize_predictions(predictions, true_labels, save_path=save_path)
    
    # Save predictions and probabilities
    results = {
        'subject_id': args.subject_id,
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist(),
        'probabilities': probabilities.tolist(),
        'metrics': metrics,
        'model_info': {
            'checkpoint': args.checkpoint,
            'model_type': args.model_type,
            'checkpoint_info': inference.checkpoint_info
        }
    }
    
    results_path = os.path.join(args.output_dir,
                               f'predictions_subject_{args.subject_id}_{timestamp}.json')
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    results = convert_to_native(results)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {results_path}")
    print(f"\nInference complete!")


if __name__ == '__main__':
    main()

    
