"""
Training script for window-adaptive sleep staging model.

Uses curriculum learning: starts with large windows, progressively trains on smaller windows.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import json
import argparse
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix
import yaml

from models.multimodal_model_crossattn_windowed import WindowAdaptiveSleepNet
from windowed_dataset import get_windowed_dataloaders


class VariableWindowTrainer:
    """Trainer with curriculum learning for variable-length windows"""
    
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"Training Window-Adaptive Model")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Output: {output_dir}\n")
        
        # Initialize model
        self.model = WindowAdaptiveSleepNet(
            n_classes=config['model']['n_classes'],
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_fusion_blocks=config['model']['n_fusion_blocks'],
            dropout=config['model']['dropout']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Mixed precision training
        self.use_amp = config['training'].get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Curriculum learning stages
        self.curriculum = config['training'].get('curriculum', {
            'stage1': {'epochs': 15, 'window_sizes': [150, 180, 200, 240]},
            'stage2': {'epochs': 15, 'window_sizes': [80, 100, 120, 150]},
            'stage3': {'epochs': 15, 'window_sizes': [40, 60, 80, 100]},
            'stage4': {'epochs': 15, 'window_sizes': [20, 30, 40, 60, 80, 100, 150]}
        })
        
        # Best model tracking
        self.best_val_kappa = 0.0
        self.best_epoch = 0
        
        # Create subdirectories
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    
    def calculate_class_weights(self, train_loader):
        """Calculate class weights from training data"""
        print("Calculating class weights...")
        class_counts = np.zeros(4)
        
        for i, batch in enumerate(train_loader):
            if len(batch) == 3:  # multimodal
                _, _, labels = batch
            else:  # ppg only
                _, labels = batch
            class_counts += np.bincount(labels.numpy().flatten(), minlength=4)
            
            if i >= 50:  # Sample first 50 batches
                break
        
        total = class_counts.sum()
        class_weights = total / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"Class counts: {class_counts}")
        print(f"Class weights: {class_weights.cpu().numpy()}\n")
        
        return class_weights
    
    def train_epoch_variable_windows(self, data_paths, window_sizes, epoch, class_weights):
        """Train one epoch with random window sampling"""
        self.model.train()
        
        # Create dataloader with random window size
        window_size = random.choice(window_sizes)
        train_loader, _, _, *datasets = get_windowed_dataloaders(
            data_path=data_paths,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            window_duration_minutes=window_size // 2,  # Convert epochs to minutes
            overlap_ratio=0.1,
            model_type='multimodal',
            use_sleepppg_test_set=True,
            pin_memory=True
        )
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        for batch_idx, (ppg, ecg, labels) in enumerate(train_loader):
            ppg = ppg.to(self.device)
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(ppg, ecg)
                    # outputs: (B, 4, n_epochs), labels: (B, n_epochs)
                    loss = self.compute_loss(outputs, labels, criterion)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(ppg, ecg)
                loss = self.compute_loss(outputs, labels, criterion)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            labs = labels.cpu().numpy()
            all_predictions.append(preds)
            all_labels.append(labs)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Window: {window_size} epochs", end='\r')
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0).flatten()
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        
        kappa = cohen_kappa_score(all_labels, all_predictions)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        avg_loss = total_loss / len(train_loader)
        
        print(f"\nEpoch {epoch} Train - Loss: {avg_loss:.4f}, "
              f"Kappa: {kappa:.4f}, Acc: {accuracy:.4f}")
        
        return avg_loss, kappa, accuracy
    
    def compute_loss(self, outputs, labels, criterion):
        """Compute loss for variable-length outputs"""
        batch_size, n_classes, seq_len = outputs.shape
        
        # Reshape for cross-entropy
        outputs_flat = outputs.transpose(1, 2).reshape(-1, n_classes)
        labels_flat = labels.reshape(-1)
        
        return criterion(outputs_flat, labels_flat)
    
    def validate(self, data_paths, window_size_minutes=100):
        """Validate on full validation set with large windows"""
        self.model.eval()
        
        _, val_loader, _, *datasets = get_windowed_dataloaders(
            data_path=data_paths,
            batch_size=1,  # Validate one subject at a time
            num_workers=0,
            window_duration_minutes=window_size_minutes,
            overlap_ratio=0.3,
            model_type='multimodal',
            use_sleepppg_test_set=True,
            pin_memory=True
        )
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for ppg, ecg, labels in val_loader:
                ppg = ppg.to(self.device)
                ecg = ecg.to(self.device)
                
                outputs = self.model(ppg, ecg)
                preds = outputs.argmax(dim=1).cpu().numpy()
                labs = labels.numpy()
                
                all_predictions.append(preds)
                all_labels.append(labs)
        
        all_predictions = np.concatenate(all_predictions, axis=0).flatten()
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        
        kappa = cohen_kappa_score(all_labels, all_predictions)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        
        return kappa, accuracy, f1_macro
    
    def train(self):
        """Main training loop with curriculum learning"""
        
        # Data paths
        data_paths = {
            'ppg': self.config['data']['ppg_path'],
            'real_ecg': self.config['data']['real_ecg_path'],
            'index': self.config['data']['index_path']
        }
        
        # Get class weights
        initial_loader, _, _, *_ = get_windowed_dataloaders(
            data_path=data_paths,
            batch_size=self.config['training']['batch_size'],
            num_workers=0,
            window_duration_minutes=60,
            model_type='multimodal'
        )
        class_weights = self.calculate_class_weights(initial_loader)
        
        # Training curriculum
        total_epochs = 0
        for stage_name, stage_config in sorted(self.curriculum.items()):
            stage_epochs = stage_config['epochs']
            window_sizes = stage_config['window_sizes']
            
            print(f"\n{'='*60}")
            print(f"{stage_name.upper()}: Training on window sizes {window_sizes} epochs")
            print(f"{'='*60}")
            
            for epoch in range(1, stage_epochs + 1):
                total_epochs += 1
                
                # Train
                train_loss, train_kappa, train_acc = self.train_epoch_variable_windows(
                    data_paths, window_sizes, total_epochs, class_weights
                )
                
                # Validate every 5 epochs
                if epoch % 5 == 0:
                    val_kappa, val_acc, val_f1 = self.validate(data_paths)
                    print(f"Validation - Kappa: {val_kappa:.4f}, "
                          f"Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
                    
                    # Update learning rate
                    self.scheduler.step(val_kappa)
                    
                    # Save best model
                    if val_kappa > self.best_val_kappa:
                        self.best_val_kappa = val_kappa
                        self.best_epoch = total_epochs
                        self.save_checkpoint('best_model.pth', total_epochs, val_kappa, val_acc)
                        print(f"✓ New best model saved (kappa: {val_kappa:.4f})")
                
                # Save periodic checkpoint
                if epoch % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{total_epochs}.pth',
                                       total_epochs, 0, 0)
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation kappa: {self.best_val_kappa:.4f} at epoch {self.best_epoch}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, filename, epoch, val_kappa, val_acc):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_kappa': val_kappa,
            'best_val_acc': val_acc,
            'config': self.config
        }, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description='Train Window-Adaptive Sleep Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(config['output']['base_dir'],
                                      f"windowed_crossattn_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    trainer = VariableWindowTrainer(config, args.output_dir)
    trainer.train()


if __name__ == '__main__':
    main()
