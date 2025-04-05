import torch
import os

def compute_delta_with_prefix_handling(base_model_path, finetune_model_path, delta_model_path):
    # Load models
    base_model = torch.load(base_model_path)
    finetune_model = torch.load(finetune_model_path)
    
    if 'model' not in base_model or 'model' not in finetune_model:
        print("Both models must have a 'model' key")
        return {}
    
    base_params = base_model['model']
    finetune_params = finetune_model['model']
    
    # Create a mapping that handles the 'xtts.' prefix
    mapping = {}
    for base_key in base_params.keys():
        # Try with the xtts prefix
        finetune_key = f"xtts.{base_key}"
        if finetune_key in finetune_params:
            mapping[base_key] = finetune_key
    
    print(f"Found {len(mapping)} matching parameters with 'xtts.' prefix")
    
    # Create delta parameters
    delta_params = {}
    total_params = 0
    changed_params = 0
    
    for base_key, finetune_key in mapping.items():
        base_tensor = base_params[base_key]
        finetune_tensor = finetune_params[finetune_key]
        
        # Verify shapes match
        if base_tensor.shape != finetune_tensor.shape:
            print(f"Warning: Shape mismatch for {base_key} -> {finetune_key}")
            continue
            
        # Calculate delta
        delta_tensor = finetune_tensor - base_tensor
        
        # Count parameters
        tensor_params = delta_tensor.numel()
        total_params += tensor_params
        
        # Check for changes
        changed_tensor = (delta_tensor.abs() > 1e-6)
        num_changed = changed_tensor.sum().item()
        changed_params += num_changed
        
        # Include if changed
        if num_changed > 0:
            delta_params[base_key] = delta_tensor
            print(f"Found changes in {base_key}: {num_changed} of {tensor_params} parameters")
    
    # Print statistics
    print(f"\nDelta computation results:")
    print(f"Total parameters mapped: {total_params}")
    if total_params > 0:
        print(f"Changed parameters: {changed_params} ({changed_params/total_params*100:.2f}%)")
    
    # Save if we have deltas
    if len(delta_params) > 0:
        save_delta(delta_params, delta_model_path)
        print(f"Saved delta model with {len(delta_params)} modified parameter groups")
    else:
        print("No meaningful delta parameters found to save")
        
    return delta_params

def save_delta(delta_state_dict, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save delta model
    torch.save(delta_state_dict, file_path)
    print(f"Delta model saved to {file_path}")

if __name__ == "__main__":
    base_checkpoint_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\base_models\v2.0.2\model.pth"
    trained_checkpoint_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\model.pth"
    delta_model_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\delta\delta.pth"
    
    # Compute delta with prefix handling
    compute_delta_with_prefix_handling(base_checkpoint_path, trained_checkpoint_path, delta_model_path)