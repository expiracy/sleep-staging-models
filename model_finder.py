import os
import json
import torch
from pathlib import Path
from datetime import datetime


def get_checkpoint_info(checkpoint_path):
    """Extract information from a checkpoint file"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_kappa': checkpoint.get('best_val_kappa', checkpoint.get('val_kappa', 'N/A')),
            'val_acc': checkpoint.get('best_val_acc', checkpoint.get('val_acc', 'N/A')),
            'test_kappa': checkpoint.get('test_kappa', 'N/A'),
            'test_acc': checkpoint.get('test_acc', 'N/A'),
        }
        
        # Format numbers
        for key in ['val_kappa', 'val_acc', 'test_kappa', 'test_acc']:
            if isinstance(info[key], float):
                info[key] = f"{info[key]:.4f}"
        
        return info
    except Exception as e:
        return {'error': str(e)}


def detect_model_type(folder_name):
    """Detect model type from folder name"""
    if 'ppg_only' in folder_name:
        return 'ppg_only'
    elif 'ppg_unfiltered' in folder_name:
        return 'ppg_unfiltered'
    elif 'crossattn_ecg' in folder_name or 'crossattn_generated' in folder_name:
        return 'crossattn_ecg'
    else:
        return 'unknown'


def get_file_size_mb(filepath):
    """Get file size in MB"""
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)


def get_modification_time(filepath):
    """Get file modification time"""
    timestamp = os.path.getmtime(filepath)
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def scan_models_directory(outputs_dir='./outputs'):
    """Scan the outputs directory for trained models"""
    
    if not os.path.exists(outputs_dir):
        print(f"Outputs directory not found: {outputs_dir}")
        return []
    
    models = []
    
    for folder in sorted(os.listdir(outputs_dir)):
        folder_path = os.path.join(outputs_dir, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Look for checkpoints directory
        checkpoints_dir = os.path.join(folder_path, 'checkpoints')
        
        if not os.path.exists(checkpoints_dir):
            continue
        
        # Check for best_model.pth
        best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
        
        if os.path.exists(best_model_path):
            model_info = {
                'folder': folder,
                'path': best_model_path,
                'type': detect_model_type(folder),
                'size_mb': get_file_size_mb(best_model_path),
                'modified': get_modification_time(best_model_path),
                'checkpoint_info': get_checkpoint_info(best_model_path)
            }
            
            # Check for config.json
            config_path = os.path.join(checkpoints_dir, 'config.json')
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        model_info['config'] = json.load(f)
                except:
                    pass
            
            models.append(model_info)
    
    return models


def print_models_table(models):
    """Print models in a formatted table"""
    
    if not models:
        print("\nNo trained models found in the outputs directory.")
        print("Train a model first using one of the training scripts.")
        return
    
    print("\n" + "="*120)
    print("AVAILABLE TRAINED MODELS")
    print("="*120)
    print(f"\nFound {len(models)} trained model(s):\n")
    
    # Group by model type
    models_by_type = {}
    for model in models:
        model_type = model['type']
        if model_type not in models_by_type:
            models_by_type[model_type] = []
        models_by_type[model_type].append(model)
    
    # Print each type
    for model_type in sorted(models_by_type.keys()):
        print(f"\n{'='*120}")
        print(f"MODEL TYPE: {model_type.upper()}")
        print(f"{'='*120}")
        
        for i, model in enumerate(models_by_type[model_type], 1):
            print(f"\n[{i}] {model['folder']}")
            print(f"    Path:     {model['path']}")
            print(f"    Size:     {model['size_mb']:.2f} MB")
            print(f"    Modified: {model['modified']}")
            
            # Print checkpoint info
            info = model['checkpoint_info']
            if 'error' not in info:
                print(f"    Epoch:    {info['epoch']}")
                if info['val_kappa'] != 'N/A':
                    print(f"    Val Kappa: {info['val_kappa']}")
                if info['val_acc'] != 'N/A':
                    print(f"    Val Acc:   {info['val_acc']}")
                if info['test_kappa'] != 'N/A':
                    print(f"    Test Kappa: {info['test_kappa']}")
                if info['test_acc'] != 'N/A':
                    print(f"    Test Acc:   {info['test_acc']}")
            
            # Usage command
            print(f"\n    Usage:")
            print(f"    python inference.py --checkpoint {model['path']} --model_type {model['type']}")
    
    print(f"\n{'='*120}\n")


def print_best_models(models):
    """Print the best performing models by type"""
    
    if not models:
        return
    
    print("\n" + "="*120)
    print("BEST PERFORMING MODELS (by Validation Kappa)")
    print("="*120)
    
    # Group by type and find best
    best_by_type = {}
    
    for model in models:
        model_type = model['type']
        val_kappa = model['checkpoint_info'].get('val_kappa', 'N/A')
        
        if val_kappa == 'N/A' or val_kappa == 'error':
            continue
        
        val_kappa_float = float(val_kappa)
        
        if model_type not in best_by_type or val_kappa_float > best_by_type[model_type]['kappa']:
            best_by_type[model_type] = {
                'model': model,
                'kappa': val_kappa_float
            }
    
    if not best_by_type:
        print("\nNo performance metrics found in checkpoints.")
        return
    
    for model_type in sorted(best_by_type.keys()):
        best = best_by_type[model_type]
        model = best['model']
        
        print(f"\n{model_type.upper()}:")
        print(f"  Folder:    {model['folder']}")
        print(f"  Path:      {model['path']}")
        print(f"  Val Kappa: {model['checkpoint_info']['val_kappa']}")
        print(f"  Val Acc:   {model['checkpoint_info']['val_acc']}")
        print(f"  Epoch:     {model['checkpoint_info']['epoch']}")
    
    print(f"\n{'='*120}\n")


def export_models_list(models, output_file='available_models.json'):
    """Export models list to JSON"""
    
    export_data = []
    
    for model in models:
        export_data.append({
            'folder': model['folder'],
            'path': model['path'],
            'type': model['type'],
            'size_mb': model['size_mb'],
            'modified': model['modified'],
            'checkpoint_info': model['checkpoint_info']
        })
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Model list exported to {output_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='List available trained models')
    parser.add_argument('--outputs_dir', type=str, default='../../outputs',
                       help='Directory containing model outputs')
    parser.add_argument('--export', type=str, default=None,
                       help='Export models list to JSON file')
    parser.add_argument('--best_only', action='store_true',
                       help='Show only best performing models')
    
    args = parser.parse_args()
    
    # Scan for models
    print(f"\nScanning directory: {args.outputs_dir}")
    models = scan_models_directory(args.outputs_dir)
    
    # Print results
    if args.best_only:
        print_best_models(models)
    else:
        print_models_table(models)
        print_best_models(models)
    
    # Export if requested
    if args.export:
        export_models_list(models, args.export)
    
    # Print summary
    if models:
        print(f"\nQuick Start:")
        print(f"  1. Choose a model from the list above")
        print(f"  2. Run inference using the provided command")
        print(f"  3. Or use run_inference_example.py and update the MODEL_CHECKPOINT variable")
        print()


if __name__ == '__main__':
    main()
