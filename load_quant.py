import torch
import os
import time

class ModelManager:
    def __init__(self, base_model_path):
        """
        Initialize the model manager with the base model
        
        Args:
            base_model_path: Path to the base model checkpoint
        """
        print(f"Loading base model from {base_model_path}")
        start_time = time.time()
        
        # Load the base model once
        self.base_checkpoint = torch.load(base_model_path, map_location="cpu")
        
        # Initialize the model using the loaded checkpoint
        self.model = self._initialize_base_model(self.base_checkpoint)
        
        print(f"Base model loaded in {time.time() - start_time:.2f} seconds")
        
    def _initialize_base_model(self, checkpoint):
        """
        Initialize the model architecture from the checkpoint.
        This will depend on your specific model architecture.
        """
        # Example initialization - adjust according to your model structure
        # For this example, we're assuming the model is stored in the 'model' key
        model = checkpoint['model']
        return model
    
    def load_quantized_state_dict(self, quant_model_path):
        """
        Load a quantized model's state dict and replace the base model's state dict
        
        Args:
            quant_model_path: Path to the quantized model checkpoint
        """
        print(f"Loading quantized state dict from {quant_model_path}")
        start_time = time.time()
        
        # Load the quantized model state dict
        quant_checkpoint = torch.load(quant_model_path, map_location="cpu")
        
        # Convert quantized state dict to compatible format for the base model
        compatible_state_dict = self._convert_quantized_to_base_format(quant_checkpoint)
        
        # Update the model with the new state dict
        self.model.load_state_dict(compatible_state_dict, strict=False)
        
        print(f"Quantized state dict loaded and applied in {time.time() - start_time:.2f} seconds")
        return self.model
    
    def _convert_quantized_to_base_format(self, quant_checkpoint):
        """
        Convert the quantized state dict format to the format expected by the base model
        
        This handles the differences between quantized and base model state dicts
        """
        # Get the quantized model state dict
        if isinstance(quant_checkpoint, dict) and 'model' in quant_checkpoint:
            quant_state_dict = quant_checkpoint['model']
        else:
            quant_state_dict = quant_checkpoint
            
        # Create a new state dict for the base model
        base_state_dict = {}
        
        # Dictionary to keep track of missing keys and unexpected keys
        missing_keys = []
        unexpected_keys = []
        
        # For each quantized parameter
        for key in quant_state_dict:
            # Check if this is a quantized parameter (with .quantized suffix)
            if key.endswith('.quantized'):
                base_key = key[:-10]  # Remove '.quantized' suffix
                scale_key = key[:-10] + '.scale'  # Get corresponding scale key
                
                if scale_key in quant_state_dict:
                    # Dequantize the parameter
                    quantized_tensor = quant_state_dict[key]
                    scale = quant_state_dict[scale_key]
                    
                    # Convert int8 to float32 and apply scale
                    dequantized_tensor = quantized_tensor.float() * scale
                    
                    # Add to base state dict
                    base_state_dict[base_key] = dequantized_tensor
                else:
                    missing_keys.append(scale_key)
            elif not key.endswith('.scale'):
                # For non-quantized parameters, copy directly
                base_state_dict[key] = quant_state_dict[key]
        
        print(f"Converted {len(base_state_dict)} parameters from quantized to base format")
        if missing_keys:
            print(f"Warning: {len(missing_keys)} scale keys were missing")
        
        return base_state_dict
    
    def get_model(self):
        """Return the current model"""
        return self.model


# Example usage
def main():
    # Paths to model checkpoints
    base_model_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\base_models\v2.0.2\model.pth"
    quant_model_path = r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\model_quantized_int8.pth"
    
    # Initialize the model manager with the base model
    manager = ModelManager(base_model_path)
    
    # Simulate multiple requests, each with a different quantized model
    for request_id in range(3):
        print(f"\nHandling request {request_id}")
        
        # Load the specific quantized state dict for this request
        # In a real scenario, you might have different quantized models
        model = manager.load_quantized_state_dict(quant_model_path)
        
        # Use the model for inference
        print(f"Model ready for inference")
        
        # Here you would run your inference code with the model
        # ...

if __name__ == "__main__":
    main()