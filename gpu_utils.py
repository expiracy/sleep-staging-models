"""
GPU memory management utilities for training
"""
import torch


def setup_gpu_memory_limit(fraction=0.8):
    """
    Limit GPU memory usage to a fraction of total available VRAM.
    
    Args:
        fraction: Fraction of total VRAM to use (default: 0.8 for 80%)
    
    This allows other applications to use the GPU while training.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory limit setup")
        return
    
    try:
        # Set memory fraction for PyTorch
        torch.cuda.set_per_process_memory_fraction(fraction)
        
        # Get total and allocated memory
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
        limited_memory = total_memory * fraction
        
        print(f"\n{'='*60}")
        print(f"GPU Memory Configuration:")
        print(f"  Device: {torch.cuda.get_device_name(device)}")
        print(f"  Total VRAM: {total_memory:.2f} GB")
        print(f"  Limited to: {limited_memory:.2f} GB ({fraction*100:.0f}%)")
        print(f"  Reserved for system: {total_memory - limited_memory:.2f} GB")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Warning: Could not set GPU memory limit: {e}")


def print_gpu_memory_usage():
    """Print current GPU memory usage statistics"""
    if not torch.cuda.is_available():
        return
    
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
    reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
    
    print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
