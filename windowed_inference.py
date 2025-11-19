"""
Windowed inference for sleep stage classification.

Performs inference on 10-hour recordings using sliding windows with overlap,
then combines predictions using weighted averaging in overlap regions.

Benefits:
- Reduced memory footprint during inference
- Supports real-time/streaming scenarios
- Compatible with existing trained models
"""

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Set thread count for inference
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Import model architectures
from models.multimodal_sleep_model import SleepPPGNet
from models.ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention
from models.multimodal_model_crossattn import ImprovedMultiModalSleepNet

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


class WindowedInference:
    """
    Perform windowed inference with overlap-based prediction fusion.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = 'ppg_only',
        device: str = 'cuda',
        window_duration_minutes: int = 5,
        overlap_ratio: float = 0.1,
        use_quantized: bool = False
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            model_type: 'ppg_only', 'ppg_unfiltered', or 'crossattn_ecg'
            device: 'cuda' or 'cpu'
            window_duration_minutes: Size of processing windows in minutes
            overlap_ratio: Overlap between consecutive windows (0.0-0.5)
            use_quantized: Whether checkpoint is quantized (forces CPU)
        """
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.window_duration_minutes = window_duration_minutes
        self.overlap_ratio = overlap_ratio
        self.use_quantized = use_quantized
        
        # Force CPU for quantized models
        if use_quantized and device == 'cuda':
            print("WARNING: Quantized models run on CPU. Switching to CPU.")
            device = 'cpu'
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Data constants
        self.sampling_rate = 34.133  # Hz
        self.epoch_duration_sec = 30
        self.samples_per_epoch = 1024
        
        # Calculate window parameters
        self.epochs_per_window = int(window_duration_minutes * 60 / self.epoch_duration_sec)
        self.samples_per_window = self.epochs_per_window * self.samples_per_epoch
        
        # Calculate stride
        overlap_epochs = int(self.epochs_per_window * overlap_ratio)
        self.stride_epochs = self.epochs_per_window - overlap_epochs
        self.stride_samples = self.stride_epochs * self.samples_per_epoch
        
        print(f"\n{'='*60}")
        print(f"Windowed Inference Configuration")
        print(f"{'='*60}")
        print(f"Device:                 {self.device}")
        print(f"Window duration:        {window_duration_minutes} min ({self.epochs_per_window} epochs)")
        print(f"Overlap ratio:          {overlap_ratio:.1%}")
        print(f"Stride:                 {self.stride_epochs} epochs")
        print(f"Samples per window:     {self.samples_per_window:,}")
        print(f"{'='*60}\n")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
    
    def _load_model(self):
        """Load model architecture and weights"""
        print(f"Loading {self.model_type} model from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Create model
        if self.model_type == 'ppg_only':
            model = SleepPPGNet()
        elif self.model_type == 'ppg_unfiltered':
            model = PPGUnfilteredCrossAttention()
        elif self.model_type == 'crossattn_ecg':
            model = ImprovedMultiModalSleepNet()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        
        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_val_kappa' in checkpoint:
            print(f"Validation kappa: {checkpoint['best_val_kappa']:.4f}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}\n")
        
        return model
    
    def _create_windows(self, signal_length: int) -> List[Tuple[int, int]]:
        """
        Create sliding window indices for a signal.
        
        Args:
            signal_length: Total length of signal in samples
        
        Returns:
            List of (start_sample, end_sample) tuples
        """
        windows = []
        start_sample = 0
        
        while start_sample + self.samples_per_window <= signal_length:
            end_sample = start_sample + self.samples_per_window
            windows.append((start_sample, end_sample))
            start_sample += self.stride_samples
        
        return windows
    
    def _process_window(
        self,
        ppg_window: np.ndarray,
        ecg_window: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process a single window through the model.
        
        Args:
            ppg_window: PPG data for this window (samples,)
            ecg_window: Optional ECG data for this window
        
        Returns:
            Class probabilities (n_classes, n_epochs_in_window)
        """
        # Convert to tensor
        ppg_tensor = torch.from_numpy(ppg_window).float().unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
        ppg_tensor = ppg_tensor.to(self.device)
        
        with torch.no_grad():
            if self.model_type == 'ppg_only':
                outputs = self.model(ppg_tensor)
            elif self.model_type == 'ppg_unfiltered':
                outputs = self.model(ppg_tensor)
            elif self.model_type == 'crossattn_ecg':
                ecg_tensor = torch.from_numpy(ecg_window).float().unsqueeze(0).unsqueeze(0)
                ecg_tensor = ecg_tensor.to(self.device)
                outputs = self.model(ppg_tensor, ecg_tensor)
        
        # outputs shape: (1, n_classes, n_epochs)
        probabilities = outputs.squeeze(0).cpu().numpy()  # (n_classes, n_epochs)
        
        return probabilities
    
    def _weighted_averaging(
        self,
        all_predictions: List[Tuple[int, int, np.ndarray]],
        total_epochs: int,
        n_classes: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine overlapping predictions using weighted averaging.
        
        Uses triangular weighting: higher weights at window center, lower at edges.
        This reduces boundary artifacts from overlapping windows.
        
        Args:
            all_predictions: List of (start_epoch, end_epoch, probabilities)
            total_epochs: Total number of epochs in full recording
            n_classes: Number of sleep stage classes
        
        Returns:
            predictions: Final predicted classes (total_epochs,)
            confidence: Prediction confidence scores (total_epochs,)
        """
        # Accumulate weighted probabilities
        prob_accumulator = np.zeros((total_epochs, n_classes))
        weight_accumulator = np.zeros(total_epochs)
        
        # Create triangular weight function (higher weight at center)
        window_weights = np.bartlett(self.epochs_per_window)  # Triangular window
        
        for start_epoch, end_epoch, probs in all_predictions:
            # probs shape: (n_classes, epochs_in_window)
            probs_t = probs.T  # (epochs_in_window, n_classes)
            
            # Apply weighted averaging
            for i, epoch_idx in enumerate(range(start_epoch, end_epoch)):
                weight = window_weights[i]
                prob_accumulator[epoch_idx] += weight * probs_t[i]
                weight_accumulator[epoch_idx] += weight
        
        # Normalize by accumulated weights (avoid division by zero)
        valid_mask = weight_accumulator > 0
        prob_accumulator[valid_mask] /= weight_accumulator[valid_mask, np.newaxis]
        
        # Get final predictions
        predictions = np.argmax(prob_accumulator, axis=1)
        confidence = np.max(prob_accumulator, axis=1)
        
        return predictions, confidence
    
    def predict_subject(
        self,
        ppg_data: np.ndarray,
        ecg_data: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict sleep stages for a full recording using windowed processing.
        
        Args:
            ppg_data: Full PPG recording (total_samples,)
            ecg_data: Optional full ECG recording
            verbose: Print progress information
        
        Returns:
            predictions: Predicted sleep stages (n_epochs,)
            confidence: Prediction confidence scores (n_epochs,)
        """
        signal_length = len(ppg_data)
        total_epochs = signal_length // self.samples_per_epoch
        
        if verbose:
            print(f"\nProcessing recording: {signal_length:,} samples ({total_epochs} epochs)")
        
        # Create sliding windows
        windows = self._create_windows(signal_length)
        
        if verbose:
            print(f"Created {len(windows)} windows")
            print(f"Processing windows...\n")
        
        # Process each window
        all_predictions = []
        
        for i, (start_sample, end_sample) in enumerate(windows):
            # Extract window data
            ppg_window = ppg_data[start_sample:end_sample]
            ecg_window = ecg_data[start_sample:end_sample] if ecg_data is not None else None
            
            # Process window
            probs = self._process_window(ppg_window, ecg_window)
            
            # Calculate epoch indices
            start_epoch = start_sample // self.samples_per_epoch
            end_epoch = start_epoch + self.epochs_per_window
            
            all_predictions.append((start_epoch, end_epoch, probs))
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(windows)} windows", end='\r')
        
        if verbose:
            print(f"  Processed {len(windows)}/{len(windows)} windows")
        
        # Combine predictions with weighted averaging
        predictions, confidence = self._weighted_averaging(
            all_predictions,
            total_epochs
        )
        
        if verbose:
            print(f"\nGenerated predictions for {len(predictions)} epochs")
            print(f"Mean confidence: {confidence.mean():.3f}")
        
        return predictions, confidence
    
    def predict_from_h5(
        self,
        h5_file_path: str,
        h5_index_file_path: str,
        subject_id: int,
        ecg_file_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load subject from H5 and predict with windowed processing.
        
        Args:
            h5_file_path: Path to PPG H5 file
            h5_index_file_path: Path to subject index H5 file
            subject_id: Subject ID to process
            ecg_file_path: Optional path to ECG H5 file
        
        Returns:
            predictions: Predicted sleep stages
            true_labels: Ground truth labels
            confidence: Prediction confidence
        """
        print(f"\nLoading subject {subject_id} from H5 files...")
        
        # Load subject indices
        with h5py.File(h5_index_file_path, 'r') as f:
            subject_key = f'{subject_id:04d}'
            
            if f'subjects/{subject_key}' not in f:
                available = list(f['subjects'].keys())[:10]
                raise ValueError(f"Subject {subject_key} not found. Available: {available}")
            
            window_indices = f[f'subjects/{subject_key}/window_indices'][:]
            start_idx = window_indices[0]
            n_windows = len(window_indices)
        
        # Load PPG data
        with h5py.File(h5_file_path, 'r') as f:
            ppg_epochs = f['ppg'][start_idx:start_idx + n_windows]
            labels = f['labels'][start_idx:start_idx + n_windows]
        
        ppg_continuous = ppg_epochs.reshape(-1)
        
        # Load ECG if needed
        ecg_continuous = None
        if self.model_type != 'ppg_only' and ecg_file_path:
            with h5py.File(ecg_file_path, 'r') as f:
                ecg_epochs = f['ecg'][start_idx:start_idx + n_windows]
            ecg_continuous = ecg_epochs.reshape(-1)
        
        print(f"Loaded {len(ppg_continuous):,} samples ({len(labels)} epochs)")
        
        # Perform windowed prediction
        predictions, confidence = self.predict_subject(
            ppg_continuous,
            ecg_continuous,
            verbose=True
        )
        
        return predictions, labels, confidence
    
    def visualize_results(
        self,
        predictions: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """Visualize predictions with optional confidence and ground truth"""
        n_plots = 1 + (true_labels is not None) + (confidence is not None)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 3 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        time_hours = np.arange(len(predictions)) * 30 / 3600
        plot_idx = 0
        
        # Plot predictions
        for i in range(len(predictions)):
            color = SLEEP_STAGE_COLORS[predictions[i]]
            axes[plot_idx].axvspan(time_hours[i], time_hours[i] + 30/3600,
                                  facecolor=color, alpha=0.7)
        
        axes[plot_idx].set_ylim(-0.5, 3.5)
        axes[plot_idx].set_yticks([0, 1, 2, 3])
        axes[plot_idx].set_yticklabels(['Wake', 'N1/N2', 'N3', 'REM'])
        axes[plot_idx].set_xlabel('Time (hours)')
        axes[plot_idx].set_ylabel('Sleep Stage')
        axes[plot_idx].set_title('Predicted Sleep Stages (Windowed Inference)')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
        
        # Plot confidence if available
        if confidence is not None:
            axes[plot_idx].plot(time_hours, confidence, linewidth=1, color='blue')
            axes[plot_idx].fill_between(time_hours, 0, confidence, alpha=0.3)
            axes[plot_idx].set_xlabel('Time (hours)')
            axes[plot_idx].set_ylabel('Confidence')
            axes[plot_idx].set_title('Prediction Confidence')
            axes[plot_idx].set_ylim([0, 1])
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot ground truth if available
        if true_labels is not None:
            valid_mask = true_labels != -1
            for i in range(len(true_labels)):
                if valid_mask[i]:
                    color = SLEEP_STAGE_COLORS[true_labels[i]]
                    axes[plot_idx].axvspan(time_hours[i], time_hours[i] + 30/3600,
                                          facecolor=color, alpha=0.7)
            
            axes[plot_idx].set_ylim(-0.5, 3.5)
            axes[plot_idx].set_yticks([0, 1, 2, 3])
            axes[plot_idx].set_yticklabels(['Wake', 'N1/N2', 'N3', 'REM'])
            axes[plot_idx].set_xlabel('Time (hours)')
            axes[plot_idx].set_ylabel('Sleep Stage')
            axes[plot_idx].set_title('Ground Truth')
            axes[plot_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def compute_metrics(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict:
        """Compute evaluation metrics"""
        from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
        
        valid_mask = true_labels != -1
        preds = predictions[valid_mask]
        labels = true_labels[valid_mask]
        
        return {
            'kappa': cohen_kappa_score(labels, preds),
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
            'f1_per_class': {
                SLEEP_STAGES[i]: f1 
                for i, f1 in enumerate(f1_score(labels, preds, average=None))
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Windowed Sleep Stage Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='ppg_only',
                        choices=['ppg_only', 'ppg_unfiltered', 'crossattn_ecg'])
    parser.add_argument('--ppg_file', type=str,
                        default='../../data/mesa_processed/mesa_ppg_with_labels.h5')
    parser.add_argument('--ecg_file', type=str,
                        default='../../data/mesa_processed/mesa_real_ecg.h5')
    parser.add_argument('--index_file', type=str,
                        default='../../data/mesa_processed/mesa_subject_index.h5')
    parser.add_argument('--subject_id', type=int, default=1)
    parser.add_argument('--window_minutes', type=int, default=5,
                        help='Window duration in minutes')
    parser.add_argument('--overlap', type=float, default=0.1,
                        help='Overlap ratio (0.0-0.5)')
    parser.add_argument('--output_dir', type=str, default='../../outputs/windowed_inference')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--quantized', action='store_true',
                        help='Use quantized model (CPU only)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize windowed inference
    inference = WindowedInference(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        window_duration_minutes=args.window_minutes,
        overlap_ratio=args.overlap,
        use_quantized=args.quantized
    )
    
    # Run prediction
    ecg_file = args.ecg_file if args.model_type != 'ppg_only' else None
    predictions, true_labels, confidence = inference.predict_from_h5(
        h5_file_path=args.ppg_file,
        h5_index_file_path=args.index_file,
        subject_id=args.subject_id,
        ecg_file_path=ecg_file
    )
    
    # Compute metrics
    metrics = inference.compute_metrics(predictions, true_labels)
    
    print(f"\n{'='*60}")
    print("EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"Cohen's Kappa:      {metrics['kappa']:.4f}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted):{metrics['f1_weighted']:.4f}")
    print("\nPer-Class F1:")
    for stage, f1 in metrics['f1_per_class'].items():
        print(f"  {stage:8s}: {f1:.4f}")
    print(f"{'='*60}\n")
    
    # Visualize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = os.path.join(args.output_dir,
                           f'windowed_hypnogram_subj{args.subject_id}_{timestamp}.png')
    inference.visualize_results(predictions, true_labels, confidence, viz_path)
    
    # Save results
    results = {
        'subject_id': args.subject_id,
        'window_duration_minutes': args.window_minutes,
        'overlap_ratio': args.overlap,
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist(),
        'confidence': confidence.tolist(),
        'metrics': metrics,
        'model_info': {
            'checkpoint': args.checkpoint,
            'model_type': args.model_type
        }
    }
    
    results_path = os.path.join(args.output_dir,
                               f'windowed_results_subj{args.subject_id}_{timestamp}.json')
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {results_path}\n")


if __name__ == '__main__':
    main()
