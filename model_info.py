import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import time


def model_keys(checkpoint_path):
    model = torch.load(checkpoint_path, map_location="cpu")  # Load on CPU to avoid GPU issues

    if isinstance(model, dict):
        print(f"Keys in model dictionary and their corresponding values:")
        for key, value in model.items():
            print(f"Key: {key}, Value shape: {value.shape if hasattr(value, 'shape') else type(value)}")
    else:
        print("The checkpoint is not a dictionary.")


def load_model(checkpoint_path, config_path):
    """Load and initialize the model with the given checkpoint, config, vocab, and speaker file."""
    start_time = time.time()

    # Load model configuration
    config = XttsConfig()
    config.load_json(config_path)


    # Initialize model and load weights
    model = Xtts.init_from_config(config)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    duration = time.time() - start_time
    print(f"Model loaded in {duration:.2f} seconds.")


    
    return model




def print_model_info(state_dict_path):
    state_dict = torch.load(state_dict_path)
    print("Type of loaded file:", type(state_dict))
    
    if isinstance(state_dict, dict):
        print("Top-level keys:", list(state_dict.keys()))
        
        # If it's a model checkpoint with 'model' key
        if 'model' in state_dict:
            print("Model key exists")
            model_dict = state_dict['model']
            print("Number of parameters in model:", len(model_dict))
            print("Sample keys:", list(model_dict.keys())[:5])  # Show first 5 keys
        
        # Check all keys for general understanding
        for key in list(state_dict.keys())[:5]:  # Limit to first 5 keys to avoid overwhelming output
            print(f"Type of {key}:", type(state_dict[key]))




def print_model_info(model):
    """Print model details including type, parameter count, and first few keys in state_dict."""
    
    state_dict = model.state_dict()
    print(f"Model type: {type(model)}")
    print(f"Number of parameters: {len(state_dict)}")

    print("First few keys in the state_dict:")
    for key in list(state_dict.keys())[:10]:
        print(key)



def print_model_weights(model):
    """Print only the weight parameters of the model."""
    state_dict = model.state_dict()

    # Iterate through the model parameters and print only the weight parameters
    print("Weights in the model:")
    for key, value in state_dict.items():
        if 'weight' in key:  # Filters out only the weights (excluding biases)
            print(f"{key}: {value.shape}")

if __name__ == "__main__":
    # Define paths to the trained model files
    trained_paths = {
        "model":r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\base_models\v2.0.2\model.pth",
        "delta": r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\delta\delta.pth",
        "checkpoint": r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\model.pth",
        "config": r"C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\config.json",
    }


    # Delta Model Logic
    print("Delta Model")
    #model_keys(trained_paths["delta"])

    trained_model = load_model( trained_paths["delta"], trained_paths["config"] )

    # Trained Model Logic
    print("Trained Model")
    #model_keys(trained_paths["checkpoint"])

    trained_model = load_model( trained_paths["checkpoint"], trained_paths["config"] )
    print_model_info(trained_model)

    print("base model")
    trained_model = load_model( trained_paths["model"], trained_paths["config"] )

    #print_model_weights(trained_model)

