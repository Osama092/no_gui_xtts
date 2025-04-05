import torch

# Path to the model checkpoint
FINETUNED_MODEL_PATH = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\model.pth"

# Load the checkpoint
checkpoint = torch.load(FINETUNED_MODEL_PATH, map_location='cpu')  # If you're using a GPU, remove 'map_location'

# Print the keys in the checkpoint to see the structure
print("Keys in checkpoint:", checkpoint.keys())

# If 'model' is a key, extract the state_dict
if 'model' in checkpoint:
    model_state_dict = checkpoint['model']
    # Show first 5 layers with tensor type, additional information, and values
    for layer_name, tensor in list(model_state_dict.items())[:5]:
        print(f"Layer: {layer_name}")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor type: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        print(f"Requires Grad: {tensor.requires_grad}")
        # Show a subset of tensor values to avoid printing too much
        print(f"Tensor values (first 5 elements): {tensor.flatten()[:5]}")  
        print("-" * 30)
