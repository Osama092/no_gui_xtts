import torch
import re

def load_and_merge_models_in_memory():
    # Hardcoded paths - replace these with your actual file paths
    BASE_MODEL_PATH = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\base_models\v2.0.2\model.pth"
    QUANTIZED_MODEL_PATH = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\model_quantized_int8.pth"
    
    print("Loading base model...")
    base_checkpoint = torch.load(BASE_MODEL_PATH, map_location='cpu')
    
    print("Loading quantized model...")
    quantized_checkpoint = torch.load(QUANTIZED_MODEL_PATH, map_location='cpu')
    
    # Extract the original state dict
    if 'state_dict' in base_checkpoint:
        base_state_dict = base_checkpoint['state_dict']
        is_nested = True
    else:
        base_state_dict = base_checkpoint
        is_nested = False
        
    quant_state_dict = quantized_checkpoint['state_dict'] if 'state_dict' in quantized_checkpoint else quantized_checkpoint
    
    # Track mapping statistics
    mapped_params = 0
    total_base_params = len(base_state_dict)
    
    print(f"Base model params: {total_base_params}")
    print(f"Quantized model params: {len(quant_state_dict)}")
    
    # First, find all parameter pairs (quantized + scale)
    quant_params = {}
    for key in quant_state_dict:
        if key.endswith('.quantized'):
            base_key = key[:-len('.quantized')]
            if base_key + '.scale' in quant_state_dict:
                quant_params[base_key] = (
                    quant_state_dict[key],
                    quant_state_dict[base_key + '.scale']
                )
    
    # Process and update the base state dict directly
    updated_keys = []
    kept_original_keys = []
    
    for key in list(base_state_dict.keys()):  # Create a copy of keys to avoid modification during iteration
        # Check if we need to map from prefix like 'xtts.' to '' (base model)
        mapped_key = None
        quant_key = None
        
        # Try direct mapping first
        if key in quant_params:
            mapped_key = key
            quant_key = key
        else:
            # Try searching for the key with potential prefix differences
            for qk in quant_params:
                # Remove potential prefix from quantized model
                pattern = r'^[^.]+\.(.*)'
                match = re.match(pattern, qk)
                if match and match.group(1) == key:
                    mapped_key = key
                    quant_key = qk
                    break
                
                # Handle case where base model has no prefix but quantized does
                match = re.match(pattern, key)
                if match and match.group(1) == qk:
                    mapped_key = key
                    quant_key = qk
                    break
        
        if mapped_key is not None:
            # Dequantize the parameter
            quantized_param, scale = quant_params[quant_key]
            
            # Handle potential shape differences
            if quantized_param.shape != base_state_dict[mapped_key].shape:
                print(f"Warning: Shape mismatch for {mapped_key}: {quantized_param.shape} vs {base_state_dict[mapped_key].shape}")
                kept_original_keys.append(key)
                continue
                
            # Dequantize int8 to float32
            if quantized_param.dtype == torch.int8:
                dequantized = quantized_param.float() * scale
                base_state_dict[mapped_key] = dequantized  # Directly update the base state dict
                mapped_params += 1
                updated_keys.append(key)
            else:
                print(f"Warning: Expected int8 for {quant_key}.quantized but got {quantized_param.dtype}")
                base_state_dict[mapped_key] = quantized_param
                updated_keys.append(key)
        elif key in quant_state_dict and not key.endswith(('.quantized', '.scale')):
            # Direct copy for non-quantized parameters
            base_state_dict[key] = quant_state_dict[key]  # Directly update the base state dict
            mapped_params += 1
            updated_keys.append(key)
        else:
            # Keep the original parameter
            kept_original_keys.append(key)
    
    print(f"Updated {len(updated_keys)}/{total_base_params} parameters")
    print(f"Kept {len(kept_original_keys)}/{total_base_params} original parameters")
    
    # At this point, base_checkpoint has been updated in-place
    # If you want to save it, uncomment the following line:
    # torch.save(base_checkpoint, "/path/to/output_model.pt")
    
    print("Base model has been updated in memory with dequantized parameters")
    return base_checkpoint  # Return the updated model

# Example usage:
if __name__ == "__main__":
    updated_model = load_and_merge_models_in_memory()
    # Now you can use updated_model directly in your code
    # For example, to move it to GPU and set to evaluation mode:
    # model = updated_model.to("cuda")
    # model.eval()