import torch

def count_zero_weights(model_path):
    """Load a model from a file and count the number of zero weights in weight tensors only."""
    model = torch.load(model_path, map_location="cpu")  # Load on CPU to avoid GPU issues
    
    print(f"Inspecting model structure: {model_path}")
    print(f"Model Type: {type(model)}")
    if isinstance(model, dict):
        print(f"Keys in model dictionary: {list(model.keys())}")
        if "model" in model:
            #print(f"Keys in 'model' dictionary: {list(model['model'].keys())}")
            state_dict = model["model"]
        else:
            state_dict = model  # Assume it's already a state_dict
    elif hasattr(model, "state_dict"):
        state_dict = model.state_dict()
    else:
        raise ValueError("Invalid model format: No valid state_dict found.")
    
    total_params = 0
    zero_params = 0

    for name, param in state_dict.items():  # Iterate over state_dict
        if "weight" in name and isinstance(param, torch.Tensor):  # Only count weights
            param_count = param.numel()
            zero_count = (param == 0).sum().item()
            
            total_params += param_count
            zero_params += zero_count
            
            # Print parameter info for debugging
            #print(f"Processing {name}: {param.shape} | Zeros: {zero_count}/{param_count} ({zero_count / param_count * 100:.2f}%)")

    sparsity = (zero_params / total_params * 100) if total_params > 0 else 0  # Avoid division by zero
    
    print(f"Model: {model_path}")
    print(f"Total Weight Parameters: {total_params}")
    print(f"Zero Weight Parameters: {zero_params}")
    print(f"Sparsity: {sparsity:.2f}%\n")
    
    return total_params, zero_params, sparsity

# Paths to your models
pruned_model_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\pruned\model_pruned.pth"
ready_model_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\model.pth"

# Check both models
count_zero_weights(pruned_model_path)
count_zero_weights(ready_model_path)
