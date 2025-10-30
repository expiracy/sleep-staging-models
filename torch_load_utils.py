"""
Utility functions for safe PyTorch model loading.
Handles PyTorch 2.6+ security requirements.
"""
import torch
import warnings

def safe_torch_load(path, map_location=None):
    """
    Safely load PyTorch checkpoint with proper configuration for PyTorch 2.6+
    
    Args:
        path: Path to the checkpoint file
        map_location: Device to map tensors to (optional)
        
    Returns:
        Loaded checkpoint
    """
    try:
        # Try loading with weights_only=False first (for models with custom objects)
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e:
        if "weights_only" in str(e):
            warnings.warn(
                f"Failed to load with weights_only=False. Error: {e}\n"
                "Attempting to load with safe globals for numpy..."
            )
            
            # Add safe globals for numpy if needed
            with torch.serialization.safe_globals([
                'numpy.core.multiarray.scalar',
                'numpy.ndarray',
                'numpy.dtype'
            ]):
                return torch.load(path, map_location=map_location, weights_only=True)
        else:
            # Re-raise if it's not a weights_only error
            raise e

def load_model_checkpoint(checkpoint_path, model, optimizer=None, map_location=None):
    """
    Load model checkpoint with proper error handling
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        map_location: Device to map tensors to (optional)
        
    Returns:
        Dictionary with loaded checkpoint data
    """
    checkpoint = safe_torch_load(checkpoint_path, map_location=map_location)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Handle cases where the checkpoint is just the model state dict
        model.load_state_dict(checkpoint)
        
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint