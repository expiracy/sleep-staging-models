"""
Resume PPG model training from a checkpoint
Simple script to continue training from checkpoint_epoch_15.pth or any checkpoint
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
from datetime import datetime
import argparse
import yaml
import gc

from models.ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention
from multimodal_dataset_aligned import get_dataloaders
from train_ppg_unfiltered import PPGUnfilteredTrainer
from torch_load_utils import safe_torch_load


def resume_training(config_path, checkpoint_path):
    """
    Resume training from a checkpoint
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to checkpoint to resume from
    """
    print("\n" + "=" * 80)
    print("RESUMING PPG TRAINING FROM CHECKPOINT")
    print("=" * 80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = safe_torch_load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"✓ Checkpoint loaded from epoch {checkpoint.get('epoch', 0)}")
    print(f"✓ Will resume training from epoch {start_epoch}")
    
    # Create trainer
    trainer = PPGUnfilteredTrainer(config, run_id=None)
    
    # Prepare data
    print("\nPreparing data...")
    data_paths = {
        'ppg': config['data']['ppg_file'],
        'index': config['data']['index_file']
    }
    
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(
        data_paths,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        model_type='ppg_only',
        use_sleepppg_test_set=True
    )
    
    # Create model
    print("\nCreating model...")
    model = PPGUnfilteredCrossAttention(
        n_classes=4,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_fusion_blocks=config['model']['n_fusion_blocks'],
        noise_config=config.get('noise_config', None)
    ).to(trainer.device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model state loaded")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Optimizer state loaded")
    
    # Create scheduler
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Load scheduler state
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✓ Scheduler state loaded")
    
    # Calculate class weights
    print("\nCalculating class weights...")
    class_weights = trainer.calculate_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    
    # Initialize training state
    best_kappa = checkpoint.get('best_overall_kappa', 0.0)
    print(f"\n✓ Previous best kappa: {best_kappa:.4f}")
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    patience = config['training']['patience']
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_overall_kappas = []
    val_median_kappas = []
    
    print(f"\n{'=' * 80}")
    print(f"STARTING TRAINING FROM EPOCH {start_epoch} TO {num_epochs}")
    print(f"{'=' * 80}\n")
    
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = trainer.train_epoch(
            model, train_loader, optimizer, criterion, scheduler, epoch
        )
        train_losses.append(train_loss)
        
        # Validate
        val_results = trainer.validate(model, val_loader, criterion)
        val_losses.append(val_results['loss'])
        val_overall_kappas.append(val_results['overall_kappa'])
        val_median_kappas.append(val_results['median_kappa'])
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f}")
        print(f"Val Overall - Acc: {val_results['overall_accuracy']:.4f}, "
              f"Kappa: {val_results['overall_kappa']:.4f}, F1: {val_results['overall_f1']:.4f}")
        print(f"Val Median  - Acc: {val_results['median_accuracy']:.4f}, "
              f"Kappa: {val_results['median_kappa']:.4f}, F1: {val_results['median_f1']:.4f}")
        
        # Print per-class performance
        stage_names = ['Wake', 'Light', 'Deep', 'REM']
        print("\nPer-class performance:")
        for i, name in enumerate(stage_names):
            print(f"  {name}: P={val_results['per_class_metrics']['precision'][i]:.3f}, "
                  f"R={val_results['per_class_metrics']['recall'][i]:.3f}, "
                  f"F1={val_results['per_class_metrics']['f1'][i]:.3f}")
        
        # Log to tensorboard
        trainer.writer.add_scalar('Train/Loss', train_loss, epoch)
        trainer.writer.add_scalar('Train/Acc', train_acc, epoch)
        trainer.writer.add_scalar('Val/Loss', val_results['loss'], epoch)
        trainer.writer.add_scalar('Val/Overall_Accuracy', val_results['overall_accuracy'], epoch)
        trainer.writer.add_scalar('Val/Overall_Kappa', val_results['overall_kappa'], epoch)
        trainer.writer.add_scalar('Val/Overall_F1', val_results['overall_f1'], epoch)
        trainer.writer.add_scalar('Val/Median_Accuracy', val_results['median_accuracy'], epoch)
        trainer.writer.add_scalar('Val/Median_Kappa', val_results['median_kappa'], epoch)
        trainer.writer.add_scalar('Val/Median_F1', val_results['median_f1'], epoch)
        trainer.writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Log per-class metrics
        for i, name in enumerate(stage_names):
            trainer.writer.add_scalar(f'Val/Precision_{name}',
                                      val_results['per_class_metrics']['precision'][i], epoch)
            trainer.writer.add_scalar(f'Val/Recall_{name}',
                                      val_results['per_class_metrics']['recall'][i], epoch)
            trainer.writer.add_scalar(f'Val/F1_{name}',
                                      val_results['per_class_metrics']['f1'][i], epoch)
        
        # Save best model (based on overall kappa)
        if val_results['overall_kappa'] > best_kappa:
            best_kappa = val_results['overall_kappa']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_overall_kappa': best_kappa,
                'best_median_kappa': val_results['median_kappa'],
                'val_acc': val_results['overall_accuracy'],
                'val_f1': val_results['overall_f1'],
                'config': config
            }
            
            best_path = os.path.join(trainer.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint_save, best_path)
            print(f"✓ Saved best model with overall kappa: {best_kappa:.4f}")
            
            # Save confusion matrix
            trainer.plot_confusion_matrix(val_results['confusion_matrix'], epoch)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch}")
            break
        
        # Periodic save
        if epoch % config['output']['save_frequency'] == 0:
            checkpoint_path_save = os.path.join(trainer.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_overall_kappa': best_kappa
            }, checkpoint_path_save)
            print(f"✓ Saved checkpoint at epoch {epoch}")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n{'=' * 80}")
    print(f"TRAINING COMPLETED")
    print(f"Best validation kappa: {best_kappa:.4f} at epoch {best_epoch}")
    print(f"{'=' * 80}\n")
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    # Load best model
    best_checkpoint = safe_torch_load(os.path.join(trainer.checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Test
    test_results = trainer.validate(model, test_loader, criterion)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_results['loss']:.4f}")
    print(f"  Overall - Acc: {test_results['overall_accuracy']:.4f}, "
          f"Kappa: {test_results['overall_kappa']:.4f}, F1: {test_results['overall_f1']:.4f}")
    print(f"  Median  - Acc: {test_results['median_accuracy']:.4f}, "
          f"Kappa: {test_results['median_kappa']:.4f}, F1: {test_results['median_f1']:.4f}")
    
    # Save results
    results = {
        'model': 'PPG + Unfiltered PPG Cross-Attention (Resumed)',
        'resumed_from_epoch': start_epoch - 1,
        'final_epoch': epoch,
        'test_accuracy_overall': test_results['overall_accuracy'],
        'test_kappa_overall': test_results['overall_kappa'],
        'test_f1_overall': test_results['overall_f1'],
        'test_accuracy_median': test_results['median_accuracy'],
        'test_kappa_median': test_results['median_kappa'],
        'test_f1_median': test_results['median_f1'],
        'test_loss': test_results['loss'],
        'best_epoch': best_epoch,
        'confusion_matrix': test_results['confusion_matrix'].tolist()
    }
    
    results_path = os.path.join(trainer.results_dir, 'resumed_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # Plot training curves
    if train_losses:
        trainer.plot_training_curves(train_losses, val_losses, val_overall_kappas, val_median_kappas)
    
    trainer.writer.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Resume PPG training from checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file to resume from')
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Resume training
    results = resume_training(args.config, args.checkpoint)
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
