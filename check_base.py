import torch
import os
from collections import defaultdict

def analyze_model_keys(model_path):
    """
    Analyze keys in a PyTorch base model and provide statistics on its parameters
    """
    print(f"Analyzing model: {os.path.basename(model_path)}")

    # Load the model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print("Successfully loaded model checkpoint")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Print top-level keys
    print(f"Type of loaded file: {type(checkpoint)}")
    print(f"Top-level keys: {list(checkpoint.keys())}")

    # Check for 'model' key
    if 'model' not in checkpoint:
        print("Error: No 'model' key found in checkpoint.")
        return

    # Get the model state dict
    state_dict = checkpoint['model']

    # Initialize counters
    total_params = len(state_dict)
    float32_params = 0
    float16_params = 0
    other_params = 0

    param_types = defaultdict(int)
    param_sizes = defaultdict(int)

    print("\nAnalyzing parameters...")
    for key, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            param_type = str(param.dtype)
            param_types[param_type] += 1
            param_sizes[key] = param.numel() * param.element_size()

            if param.dtype == torch.float32:
                float32_params += 1
            elif param.dtype == torch.float16:
                float16_params += 1
            else:
                other_params += 1
        else:
            param_types["non-tensor"] += 1
            other_params += 1

    # Print statistics
    print(f"\nTotal parameters: {total_params}")
    print(f"Float32 parameters: {float32_params}")
    print(f"Float16 parameters: {float16_params}")
    print(f"Other parameter types: {other_params}")

    # Print parameter type distribution
    print("\nParameter type distribution:")
    for param_type, count in param_types.items():
        print(f"  {param_type}: {count}")

    # Calculate and print total memory usage
    total_bytes = sum(param_sizes.values())
    print(f"\nTotal memory usage: {total_bytes / (1024*1024):.2f} MB")

    # Print example parameter keys
    print("\nExample parameter keys:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        if isinstance(state_dict[key], torch.Tensor):
            print(f"  {key}: {state_dict[key].dtype}, shape={state_dict[key].shape}")
        else:
            print(f"  {key}: (not a tensor)")

    if len(state_dict) > 10:
        print(f"  ... and {len(state_dict) - 10} more parameters")

if __name__ == "__main__":
    model_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\base_models\v2.0.2\model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
    else:
        analyze_model_keys(model_path)
