import torch
import os
import numpy as np

def load_model(checkpoint_path):
    """Load a model from a checkpoint file."""
    print(f"Loading model from {checkpoint_path}")
    return torch.load(checkpoint_path)

def verify_delta(base_model_path, finetune_model_path, delta_model_path):
    """Verify the delta model by applying it to the base model and comparing with the fine-tuned model."""
    # Load all models
    base_model = load_model(base_model_path)
    finetune_model = load_model(finetune_model_path)
    delta_model = load_model(delta_model_path)
    
    # Check if model exists in all dictionaries
    if 'model' not in base_model or 'model' not in finetune_model:
        print("Error: Both base and fine-tuned models must have a 'model' key")
        return False
    
    base_params = base_model['model']
    finetune_params = finetune_model['model']
    
    # Create a reconstructed model by applying delta to base model
    reconstructed_params = {}
    
    # Keep track of statistics
    total_params_checked = 0
    matching_params = 0
    minor_diff_params = 0
    major_diff_params = 0
    
    # Going through each base parameter
    print("\nVerifying parameters:")
    for base_key, base_tensor in base_params.items():
        # The key in the fine-tuned model has 'xtts.' prefix
        finetune_key = f"xtts.{base_key}"
        
        if finetune_key in finetune_params:
            # Get corresponding tensors
            finetune_tensor = finetune_params[finetune_key]
            
            # Apply delta if available
            if base_key in delta_model:
                reconstructed_tensor = base_tensor + delta_model[base_key]
            else:
                reconstructed_tensor = base_tensor
            
            # Calculate the difference between reconstructed and fine-tuned tensor
            diff = (reconstructed_tensor - finetune_tensor).abs()
            max_diff = diff.max().item()
            
            # Count parameters
            num_params = finetune_tensor.numel()
            total_params_checked += num_params
            
            # Evaluate the difference
            if max_diff < 1e-6:
                matching_params += num_params
                status = "MATCH"
            elif max_diff < 1e-3:
                minor_diff_params += num_params
                status = "MINOR DIFF"
            else:
                major_diff_params += num_params
                status = "MAJOR DIFF"
            
            # Print for a sample of parameters
            if np.random.random() < 0.01:  # Print ~1% of parameters
                print(f"{base_key}: {status} (max diff: {max_diff:.6f})")
        else:
            print(f"Warning: {base_key} has no corresponding key in fine-tuned model")
    
    # Calculate overall accuracy
    accuracy = 0
    if total_params_checked > 0:
        accuracy = (matching_params + minor_diff_params) / total_params_checked * 100
    
    # Print summary statistics
    print("\nVerification Results:")
    print(f"Total parameters checked: {total_params_checked}")
    print(f"Exact matches: {matching_params} ({matching_params/total_params_checked*100:.2f}%)")
    print(f"Minor differences (<1e-3): {minor_diff_params} ({minor_diff_params/total_params_checked*100:.2f}%)")
    print(f"Major differences: {major_diff_params} ({major_diff_params/total_params_checked*100:.2f}%)")
    print(f"Overall accuracy: {accuracy:.2f}%")
    
    # Determine if verification passed
    passed = major_diff_params == 0 and (matching_params + minor_diff_params) / total_params_checked > 0.99
    if passed:
        print("\nDelta verification PASSED! The delta correctly transforms the base model to match the fine-tuned model.")
    else:
        print("\nDelta verification FAILED! The delta does not correctly transform the base model.")
    
    return passed

if __name__ == "__main__":
    base_checkpoint_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\base_models\v2.0.2\model.pth"
    trained_checkpoint_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\model.pth"
    delta_model_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\delta\delta.pth"
    
    # Verify the delta model
    verify_delta(base_checkpoint_path, trained_checkpoint_path, delta_model_path)