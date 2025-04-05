import torch
import os
from collections import defaultdict

def analyze_model_keys(model_path):
    """
    Analyze keys in a PyTorch model and provide statistics on quantized parameters
    """
    print(f"Analyzing model: {os.path.basename(model_path)}")

    # Load the model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print("Successfully loaded model checkpoint")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Check if model exists in checkpoint
    if 'model' not in checkpoint:
        print("Warning: No 'model' key found in checkpoint.")
        print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
        return

    # Get the model state dict
    state_dict = checkpoint['model']

    # Print quantization info if available
    if 'quantization_info' in checkpoint:
        print(f"Quantization info: {checkpoint['quantization_info']}")

    # Initialize counters
    total_params = len(state_dict)
    quantized_params = 0
    scale_params = 0
    float32_params = 0
    float16_params = 0
    other_params = 0

    # Track parameter types and shapes
    param_types = defaultdict(int)
    param_sizes = defaultdict(int)
    quantized_suffixes = ['.quantized', '.scale']

    # Analyze each parameter
    print("\nAnalyzing parameters...")
    for key, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            param_type = str(param.dtype)
            param_types[param_type] += 1
            param_sizes[key] = param.numel() * param.element_size()

            if key.endswith('.quantized'):
                quantized_params += 1
            elif key.endswith('.scale'):
                scale_params += 1
            elif param.dtype == torch.float32:
                float32_params += 1
            elif param.dtype == torch.float16:
                float16_params += 1
            else:
                other_params += 1
        else:
            param_types["non-tensor"] += 1
            other_params += 1

    # Find base parameter names (without quantization suffixes)
    base_params = set()
    quantized_base_params = set()

    for key in state_dict.keys():
        base_name = key
        for suffix in quantized_suffixes:
            if key.endswith(suffix):
                base_name = key[:-len(suffix)]
                quantized_base_params.add(base_name)
                break
        base_params.add(base_name)

    # Calculate statistics
    unique_base_params = len(base_params)
    quantized_base_count = len(quantized_base_params)
    non_quantized_count = unique_base_params - quantized_base_count

    # Print statistics
    print(f"\nTotal parameters: {total_params}")
    print(f"Unique base parameters: {unique_base_params}")
    print(f"Quantized parameters (with .quantized suffix): {quantized_params}")
    print(f"Scale parameters (with .scale suffix): {scale_params}")
    print(f"Float32 parameters: {float32_params}")
    print(f"Float16 parameters: {float16_params}")
    print(f"Other parameter types: {other_params}")
    print(f"Base parameters that are quantized: {quantized_base_count}")
    print(f"Base parameters that are not quantized: {non_quantized_count}")

    # Print parameter type distribution
    print("\nParameter type distribution:")
    for param_type, count in param_types.items():
        print(f"  {param_type}: {count}")

    # Calculate and print total memory usage
    total_bytes = sum(param_sizes.values())
    print(f"\nTotal memory usage: {total_bytes / (1024*1024):.2f} MB")

    # Print the first few parameter keys as examples
    print("\nExample parameter keys:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        if isinstance(state_dict[key], torch.Tensor):
            print(f"  {key}: {state_dict[key].dtype}, shape={state_dict[key].shape}")
        else:
            print(f"  {key}: (not a tensor)")

    if len(state_dict) > 10:
        print(f"  ... and {len(state_dict) - 10} more parameters")

if __name__ == "__main__":
    model_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\model_quantized_int8.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
    else:
        analyze_model_keys(model_path)
