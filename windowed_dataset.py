"""
Windowed dataset for memory-efficient sleep stage classification.

Processes 10-hour PPG recordings in smaller windows with overlap,
reducing memory footprint during training and inference.

Key Features:
- Processes N-minute windows instead of full 10-hour sequences
- Configurable overlap for smooth predictions
- Compatible with existing model architectures
- Supports both training and inference modes
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional, List

# Import the test set from original dataset
from multimodal_dataset_aligned import SLEEPPPG_TEST_SUBJECTS


class WindowedSleepDataset(Dataset):
    """
    Dataset that yields windows of PPG/ECG data instead of full sequences.
    
    Key parameters:
    - window_duration_minutes: Size of each window (e.g., 5 minutes)
    - overlap_ratio: Overlap between consecutive windows (0.0 to 0.5 recommended)
    - stride_windows: Number of 30-sec epochs to advance per window (calculated from overlap)
    """
    
    def __init__(
        self, 
        data_path: Dict[str, str],
        split: str = 'train',
        window_duration_minutes: int = 5,
        overlap_ratio: float = 0.1,
        use_generated_ecg: bool = False,
        model_type: str = 'ppg_only',
        transform=None,
        seed: int = 42,
        use_sleepppg_test_set: bool = True
    ):
        """
        Args:
            data_path: Dict with keys 'ppg', 'ecg'/'real_ecg', 'index'
            split: 'train', 'val', or 'test'
            window_duration_minutes: Duration of each window in minutes
            overlap_ratio: Fraction of overlap between windows (0.0-0.5)
            use_generated_ecg: Use synthetic ECG if True
            model_type: 'ppg_only', 'multimodal'
            transform: Optional data augmentation
            seed: Random seed for splits
            use_sleepppg_test_set: Use predefined test set from SleepPPG-Net
        """
        self.split = split
        self.window_duration_minutes = window_duration_minutes
        self.overlap_ratio = overlap_ratio
        self.use_generated_ecg = use_generated_ecg
        self.model_type = model_type
        self.transform = transform
        self.seed = seed
        self.use_sleepppg_test_set = use_sleepppg_test_set
        
        # Data constants
        self.sampling_rate = 34.133  # Hz
        self.epoch_duration_sec = 30
        self.samples_per_epoch = 1024
        self.epochs_per_subject = 1200  # 10 hours
        self.total_samples_per_subject = self.epochs_per_subject * self.samples_per_epoch
        
        # Calculate window parameters
        self.epochs_per_window = int(window_duration_minutes * 60 / self.epoch_duration_sec)
        self.samples_per_window = self.epochs_per_window * self.samples_per_epoch
        
        # Calculate stride (how many epochs to advance per window)
        overlap_epochs = int(self.epochs_per_window * overlap_ratio)
        self.stride_epochs = self.epochs_per_window - overlap_epochs
        self.stride_samples = self.stride_epochs * self.samples_per_epoch
        
        # Number of windows per subject
        self.windows_per_subject = (self.epochs_per_subject - self.epochs_per_window) // self.stride_epochs + 1
        
        print(f"\n{'='*60}")
        print(f"Windowed Dataset Configuration")
        print(f"{'='*60}")
        print(f"Window duration:        {window_duration_minutes} minutes ({self.epochs_per_window} epochs)")
        print(f"Samples per window:     {self.samples_per_window:,}")
        print(f"Overlap ratio:          {overlap_ratio:.1%}")
        print(f"Stride:                 {self.stride_epochs} epochs")
        print(f"Windows per subject:    {self.windows_per_subject}")
        print(f"Memory reduction:       {self.total_samples_per_subject / self.samples_per_window:.1f}x per batch")
        print(f"{'='*60}\n")
        
        # Load file paths
        self.ppg_file_path = data_path['ppg']
        self.ecg_file_path = data_path.get('ecg', data_path.get('real_ecg'))
        self.index_file_path = data_path['index']
        
        # Verify files exist
        for path in [self.ppg_file_path, self.index_file_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
        
        if model_type != 'ppg_only' and not os.path.exists(self.ecg_file_path):
            raise FileNotFoundError(f"ECG file not found: {self.ecg_file_path}")
        
        # Prepare subject splits
        self._prepare_subjects()
        
        # Build window index
        self._build_window_index()
    
    def _prepare_subjects(self):
        """
        Split subjects into train/val/test sets.
        
        CRITICAL: Split is done at SUBJECT level to prevent data leakage.
        All windows from a subject go to the same split (train/val/test).
        This ensures the model never sees overlapping/adjacent windows from 
        validation subjects during training.
        """
        with h5py.File(self.index_file_path, 'r') as f:
            all_subjects = list(f['subjects'].keys())
            
            # Filter valid subjects (must have exactly 1200 epochs)
            valid_subjects = []
            for subj in all_subjects:
                n_windows = f[f'subjects/{subj}'].attrs['n_windows']
                if n_windows == self.epochs_per_subject:
                    valid_subjects.append(subj)
        
        if self.use_sleepppg_test_set:
            # Use predefined test set from SleepPPG-Net paper
            test_subjects = [s for s in SLEEPPPG_TEST_SUBJECTS if s in valid_subjects]
            train_val_subjects = [s for s in valid_subjects if s not in test_subjects]
            
            # Split train_val into train and val at SUBJECT level
            # This ensures no subject appears in both train and val
            train_subjects, val_subjects = train_test_split(
                train_val_subjects, test_size=0.2, random_state=self.seed
            )
        else:
            # Random split at subject level
            train_subjects, test_subjects = train_test_split(
                valid_subjects, test_size=0.2, random_state=self.seed
            )
            train_subjects, val_subjects = train_test_split(
                train_subjects, test_size=0.2, random_state=self.seed
            )
        
        # Select subjects for this split
        if self.split == 'train':
            self.subjects = train_subjects
        elif self.split == 'val':
            self.subjects = val_subjects
        else:  # test
            self.subjects = test_subjects
        
        print(f"{self.split} set: {len(self.subjects)} subjects")
        print(f"  Each subject contributes {self.windows_per_subject} windows")
        print(f"  Total windows: {len(self.subjects) * self.windows_per_subject}")
        
        # Load subject start indices from H5
        self.subject_start_indices = {}
        with h5py.File(self.index_file_path, 'r') as f:
            for subj in self.subjects:
                indices = f[f'subjects/{subj}/window_indices'][:]
                if len(indices) == self.epochs_per_subject:
                    self.subject_start_indices[subj] = indices[0]
    
    def _build_window_index(self):
        """
        Build an index mapping dataset index to (subject_id, window_start_epoch).
        This allows efficient random access to any window.
        """
        self.window_index = []
        
        for subject_id in self.subjects:
            for window_idx in range(self.windows_per_subject):
                start_epoch = window_idx * self.stride_epochs
                self.window_index.append((subject_id, start_epoch))
        
        print(f"Total windows in {self.split} set: {len(self.window_index)}")
    
    def __len__(self):
        return len(self.window_index)
    
    def __getitem__(self, idx):
        """
        Returns a single window of data.
        
        Returns:
            For PPG-only models: (ppg_window, labels_window)
            For multimodal models: (ppg_window, ecg_window, labels_window)
            
            Shapes:
                ppg_window: (1, samples_per_window)
                ecg_window: (1, samples_per_window) 
                labels_window: (epochs_per_window,)
        """
        subject_id, start_epoch = self.window_index[idx]
        
        # Calculate indices in H5 file
        subject_base_idx = self.subject_start_indices[subject_id]
        h5_start_epoch = subject_base_idx + start_epoch
        h5_end_epoch = h5_start_epoch + self.epochs_per_window
        
        # Load data from H5 files
        with h5py.File(self.ppg_file_path, 'r') as f:
            ppg_epochs = f['ppg'][h5_start_epoch:h5_end_epoch]  # (epochs_per_window, 1024)
            labels = f['labels'][h5_start_epoch:h5_end_epoch]   # (epochs_per_window,)
        
        # Reshape to continuous signal
        ppg_continuous = ppg_epochs.reshape(-1)  # (samples_per_window,)
        
        if self.model_type != 'ppg_only':
            with h5py.File(self.ecg_file_path, 'r') as f:
                ecg_epochs = f['ecg'][h5_start_epoch:h5_end_epoch]
            ecg_continuous = ecg_epochs.reshape(-1)
            
            # Apply transforms if provided
            if self.transform:
                ppg_continuous, ecg_continuous = self.transform(ppg_continuous, ecg_continuous)
            
            # Convert to tensors
            ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)  # (1, samples)
            ecg_tensor = torch.FloatTensor(ecg_continuous).unsqueeze(0)
            labels_tensor = torch.LongTensor(labels)
            
            return ppg_tensor, ecg_tensor, labels_tensor
        else:
            # PPG-only
            if self.transform:
                ppg_continuous = self.transform(ppg_continuous)
            
            ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)
            labels_tensor = torch.LongTensor(labels)
            
            return ppg_tensor, labels_tensor


def get_windowed_dataloaders(
    data_path: Dict[str, str],
    batch_size: int = 4,
    num_workers: int = 4,
    window_duration_minutes: int = 5,
    overlap_ratio: float = 0.1,
    use_generated_ecg: bool = False,
    model_type: str = 'ppg_only',
    use_sleepppg_test_set: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, WindowedSleepDataset, WindowedSleepDataset, WindowedSleepDataset]:
    """
    Factory function to create train/val/test dataloaders with windowed data.
    
    Args:
        data_path: Dict with paths to H5 files
        batch_size: Batch size (can be larger than 1 now!)
        num_workers: Number of data loading workers
        window_duration_minutes: Size of each window in minutes
        overlap_ratio: Overlap between consecutive windows (0.0-0.5)
        use_generated_ecg: Use synthetic ECG
        model_type: 'ppg_only' or 'multimodal'
        use_sleepppg_test_set: Use SleepPPG-Net test set
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    train_dataset = WindowedSleepDataset(
        data_path=data_path,
        split='train',
        window_duration_minutes=window_duration_minutes,
        overlap_ratio=overlap_ratio,
        use_generated_ecg=use_generated_ecg,
        model_type=model_type,
        use_sleepppg_test_set=use_sleepppg_test_set
    )
    
    val_dataset = WindowedSleepDataset(
        data_path=data_path,
        split='val',
        window_duration_minutes=window_duration_minutes,
        overlap_ratio=overlap_ratio,
        use_generated_ecg=use_generated_ecg,
        model_type=model_type,
        use_sleepppg_test_set=use_sleepppg_test_set
    )
    
    test_dataset = WindowedSleepDataset(
        data_path=data_path,
        split='test',
        window_duration_minutes=window_duration_minutes,
        overlap_ratio=overlap_ratio,
        use_generated_ecg=use_generated_ecg,
        model_type=model_type,
        use_sleepppg_test_set=use_sleepppg_test_set
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    """Test windowed dataset"""
    
    # Example configuration
    data_paths = {
        'ppg': "../../data/mesa_processed/mesa_ppg_with_labels.h5",
        'real_ecg': "../../data/mesa_processed/mesa_real_ecg.h5",
        'index': "../../data/mesa_processed/mesa_subject_index.h5"
    }
    
    print("Testing Windowed Dataset\n")
    
    # Test different window sizes
    for window_minutes in [3, 5, 10]:
        print(f"\n{'='*60}")
        print(f"Testing {window_minutes}-minute windows")
        print(f"{'='*60}")
        
        train_loader, val_loader, test_loader, *datasets = get_windowed_dataloaders(
            data_path=data_paths,
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            window_duration_minutes=window_minutes,
            overlap_ratio=0.1,
            model_type='ppg_only'
        )
        
        # Test loading a few batches
        print(f"\nLoading sample batches...")
        train_dataset_obj = datasets[0]  # First dataset returned
        for i, batch in enumerate(train_loader):
            if train_dataset_obj.model_type == 'ppg_only':
                ppg, labels = batch
                print(f"Batch {i+1}: PPG shape {ppg.shape}, Labels shape {labels.shape}")
            else:
                ppg, ecg, labels = batch
                print(f"Batch {i+1}: PPG {ppg.shape}, ECG {ecg.shape}, Labels {labels.shape}")
            
            if i >= 2:  # Just test a few batches
                break
        
        print(f"\nTotal batches: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test")
