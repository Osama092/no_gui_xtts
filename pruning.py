import torch
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import time
import os
import shutil

def calculate_dynamic_threshold(weight_tensor, max_pruning_rate=0.15, importance_factor=0.7):
    """
    Calculate a dynamic pruning threshold based on weight tensor statistics,
    with a conservative maximum pruning rate to avoid audio quality issues.
    
    Args:
        weight_tensor: The weight tensor to analyze
        max_pruning_rate: Maximum allowed pruning percentage (0.0 to 1.0)
        importance_factor: Factor to adjust threshold based on layer importance (0.0 to 1.0)
            Higher values make the threshold more sensitive to layer importance
    
    Returns:
        Dynamic pruning threshold amount for this specific tensor
    """
    # Calculate statistics of the weight tensor
    abs_weights = torch.abs(weight_tensor)
    mean_val = torch.mean(abs_weights).item()
    std_val = torch.std(abs_weights).item()
    
    # Calculate weight sparsity (naturally occurring zeros)
    natural_sparsity = (weight_tensor == 0).float().mean().item()
    
    # Calculate weight density (ratio of non-zero to total elements)
    density = 1.0 - natural_sparsity
    
    # Calculate coefficient of variation (measure of weight distribution)
    cv = std_val / (mean_val + 1e-10)  # Adding small epsilon to avoid division by zero
    
    # Calculate magnitude spread (difference between max and mean)
    max_val = torch.max(abs_weights).item()
    magnitude_spread = max_val / (mean_val + 1e-10)
    
    # Analyze weight distribution quartiles
    q75 = torch.quantile(abs_weights, 0.75).item()
    q25 = torch.quantile(abs_weights, 0.25).item()
    iqr = q75 - q25
    
    # Importance indicators:
    # 1. Higher density = more important (fewer natural zeros)
    # 2. Higher CV = more varied weights = likely more important
    # 3. Higher magnitude spread = some critical weights = important
    # 4. Higher IQR = wider distribution = might be important
    importance_score = (density + cv + magnitude_spread/10 + iqr/(max_val+1e-10))/4
    
    # More important layers get pruned less (inversely proportional)
    # Start with a very conservative base rate
    base_rate = max_pruning_rate * 0.5
    
    # Calculate dynamic threshold - scale based on importance
    dynamic_amount = base_rate * (1 - importance_factor * importance_score)
    
    # Ensure amount stays within reasonable bounds
    # Much lower upper bound to prevent audio quality issues
    min_rate = 0.01  # At minimum, prune 1% of weights
    dynamic_amount = max(min_rate, min(max_pruning_rate, dynamic_amount))
    
    return dynamic_amount

def manual_pruning(model, method="l1", amount=0.1, dynamic_threshold=True):
    """
    Apply manual pruning to the model.
    
    Args:
        model: The PyTorch model to prune
        method: The pruning method to use ('l1', 'random')
        amount: The base amount/percentage of parameters to prune (0.0 to 1.0)
        dynamic_threshold: Whether to use dynamic thresholding instead of fixed amount
    
    Returns:
        Pruned model
    """
    print(f"Applying {method} pruning with {'dynamic' if dynamic_threshold else 'fixed'} threshold (max amount: {amount})")
    
    total_params = 0
    pruned_params = 0
    
    # Track pruning statistics across layers
    layer_stats = {}
    
    # First pass: analyze all layers to understand their importance
    if dynamic_threshold:
        layer_importances = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv1d) or \
               isinstance(module, torch.nn.Conv2d):
                weight = module.weight.data
                layer_importances[name] = calculate_dynamic_threshold(weight, max_pruning_rate=amount)
    
    # Second pass: actual pruning with adjusted thresholds
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv1d) or \
           isinstance(module, torch.nn.Conv2d):
            
            # Get the weight tensor
            weight = module.weight.data
            total_params += weight.numel()
            
            # Skip tiny layers or embedding layers (often critical for performance)
            if "embed" in name.lower() or weight.numel() < 1000:
                print(f"Skipping pruning for critical or small layer: {name}")
                continue
            
            # Determine pruning amount for this layer
            layer_amount = amount
            if dynamic_threshold:
                layer_amount = layer_importances[name]
                print(f"Layer {name}: Using dynamic pruning amount of {layer_amount:.4f}")
            
            # Create a mask for pruning
            mask = torch.ones_like(weight)
            
            if method == "l1":
                # Get the absolute values
                abs_weights = torch.abs(weight).flatten()
                k = int(layer_amount * abs_weights.shape[0])
                if k > 0:
                    threshold = torch.kthvalue(abs_weights, k).values
                    mask = torch.where(torch.abs(weight) > threshold, torch.ones_like(weight), torch.zeros_like(weight))
            
            elif method == "random":
                # Generate random mask
                rand_mask = torch.rand_like(weight) > layer_amount
                mask = rand_mask.float()
            
            # Apply mask to weights
            pruned_weights = weight * mask
            layer_pruned = weight.numel() - torch.sum(mask).item()
            pruned_params += layer_pruned
            
            # Track statistics for this layer
            layer_stats[name] = {
                "total": weight.numel(),
                "pruned": layer_pruned,
                "pruned_pct": 100 * layer_pruned / weight.numel()
            }
            
            # Update the weights
            module.weight.data = pruned_weights
    
    # Print overall statistics
    print(f"Pruned {pruned_params}/{total_params} parameters ({100*pruned_params/total_params:.2f}%)")
    
    # Print the most and least pruned layers for analysis
    if layer_stats:
        sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]["pruned_pct"], reverse=True)
        print("\nTop 5 most pruned layers:")
        for name, stats in sorted_layers[:5]:
            print(f"  {name}: {stats['pruned_pct']:.2f}% ({stats['pruned']}/{stats['total']})")
        
        print("\nTop 5 least pruned layers:")
        for name, stats in sorted_layers[-5:]:
            print(f"  {name}: {stats['pruned_pct']:.2f}% ({stats['pruned']}/{stats['total']})")
    
    return model

def save_pruned_model(model, output_dir, config_path, original_checkpoint_path):
    """
    Save the pruned model and config file to a separate directory, including all necessary information.
    
    Args:
        model: The pruned PyTorch model to save
        output_dir: Directory to save the pruned model
        config_path: Path to the original config file
        original_checkpoint_path: Path to the original model checkpoint (for its keys)
    
    Returns:
        Path to the saved pruned model checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original model checkpoint to extract its structure
    original_checkpoint = torch.load(original_checkpoint_path, map_location=torch.device("cpu"))
    
    # Save model weights along with the same keys as the original checkpoint
    pruned_model_path = os.path.join(output_dir, "model_pruned.pth")
    
    # Prepare the new checkpoint dictionary similar to the original model
    checkpoint = {
        'model': model.state_dict(),
        'config': config_path,
        'scaler': original_checkpoint.get('scaler', None),
        'step': original_checkpoint.get('step', None),
        'epoch': original_checkpoint.get('epoch', None),
        'date': original_checkpoint.get('date', None),
        'model_loss': original_checkpoint.get('model_loss', None)
    }
    
    # Save the full checkpoint
    torch.save(checkpoint, pruned_model_path)
    
    # Copy the config file (required for model initialization)
    shutil.copy(config_path, os.path.join(output_dir, "config.json"))
    
    print(f"Pruned model and config saved to {output_dir}")
    return pruned_model_path


def load_model(checkpoint_path, config_path, apply_pruning=False, pruning_method="l1", pruning_amount=0.1, dynamic_threshold=True):
    """Load and initialize the model with the given checkpoint and config."""
    start_time = time.time()

    # Load model configuration
    config = XttsConfig()
    config.load_json(config_path)
    
    # Initialize model and load weights
    model = Xtts.init_from_config(config)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    
    # Apply pruning if requested
    if apply_pruning:
        model = manual_pruning(model, method=pruning_method, amount=pruning_amount, dynamic_threshold=dynamic_threshold)

    duration = time.time() - start_time
    print(f"Model loaded in {duration:.2f} seconds.")
    
    return model

if __name__ == "__main__":
    # Define paths to the trained model files
    trained_paths = {
        "checkpoint": r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\model.pth",
        "config": r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\config.json",
    }
    
    # Define output directory for the pruned model
    pruned_model_dir = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\pruned"

    # Load the trained model with pruning
    trained_model = load_model(
        trained_paths["checkpoint"], 
        trained_paths["config"],
        apply_pruning=True, 
        pruning_method="l1", 
        pruning_amount=0.0000001,  # Reduced from 0.3 to 0.1 as max pruning rate
        dynamic_threshold=True  # Enable dynamic threshold
    )
    
    # Save the pruned model to a separate directory
    pruned_model_path = save_pruned_model(
        trained_model, 
        pruned_model_dir,
        trained_paths["config"],
        trained_paths["checkpoint"]
    )
    
    print(f"Pruned model saved to: {pruned_model_path}")