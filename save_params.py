import torch
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def load_model(checkpoint_path, config_path, vocab_path, speaker_file_path):
    # Load model configuration
    config = XttsConfig()
    config.load_json(config_path)
    
    # Initialize model
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=checkpoint_path, vocab_path=vocab_path, speaker_file_path=speaker_file_path, use_deepspeed=False)
    
    if torch.cuda.is_available():
        model.cuda()
    
    return model

def calculate_delta(base_model, new_model):
    # Assuming models are compatible
    base_state_dict = base_model.state_dict()
    new_state_dict = new_model.state_dict()

    # Calculate delta by subtracting base model parameters from new model parameters
    delta_state_dict = {}
    for key in new_state_dict:
        if key in base_state_dict:
            delta_state_dict[key] = new_state_dict[key] - base_state_dict[key]
        else:
            raise KeyError(f"Key {key} found in new model but not in base model")

    return delta_state_dict

if __name__ == "__main__":
    base_checkpoint_path = r"C:\Users\mohamed aziz\Desktop\pfe\VoiceCloning\xtts-finetune-webui\base_models\v2.0.2\model.pth"
    config_path = r"C:\Users\mohamed aziz\Desktop\pfe\VoiceCloning\xtts-finetune-webui\base_models\v2.0.2\config.json"
    vocab_path = r"C:\Users\mohamed aziz\Desktop\pfe\VoiceCloning\xtts-finetune-webui\base_models\v2.0.2\vocab.json"
    speaker_file_path = r"C:\Users\mohamed aziz\Desktop\pfe\VoiceCloning\xtts-finetune-webui\base_models\v2.0.2\speakers_xtts.pth"

    new_checkpoint_path = r"C:\Users\mohamed aziz\Desktop\pfe\VoiceCloning\xtts-webui\models\mike\model.pth"
    config_path = r"C:\Users\mohamed aziz\Desktop\pfe\VoiceCloning\xtts-webui\models\mike\config.json"
    vocab_path = r"C:\Users\mohamed aziz\Desktop\pfe\VoiceCloning\xtts-webui\models\mike\vocab.json"
    speaker_file_path = r"C:\Users\mohamed aziz\Desktop\pfe\VoiceCloning\xtts-webui\models\mike\speakers_xtts.pth"

    base_model = load_model(base_checkpoint_path, config_path, vocab_path, speaker_file_path)
    new_model = load_model(new_checkpoint_path, config_path, vocab_path, speaker_file_path)

    base_state_dict = base_model.state_dict()
    new_state_dict = new_model.state_dict()

    delta_state_dict = calculate_delta(base_model, new_model)

    print("Parameters with different precision or extra metadata:")
    
    for key in delta_state_dict:
        if key in base_state_dict:
            base_tensor = base_state_dict[key]
            delta_tensor = delta_state_dict[key]

            # Check precision (dtype)
            if base_tensor.dtype != delta_tensor.dtype:
                print(f"Key: {key}")
                print(f"Base model dtype: {base_tensor.dtype}")
                print(f"Delta model dtype: {delta_tensor.dtype}")
                print("---")
                
            # Check for extra metadata or additional data
            if base_tensor.shape != delta_tensor.shape:
                print(f"Key: {key}")
                print(f"Base model shape: {base_tensor.shape}")
                print(f"Delta model shape: {delta_tensor.shape}")
                print("---")

    # Inspect file sizes manually
    base_model_size = os.path.getsize(base_checkpoint_path) / (1024 * 1024 * 1024)  # Size in GB
    delta_model_size = os.path.getsize(new_checkpoint_path) / (1024 * 1024 * 1024)  # Size in GB

    print(f"Base model size: {base_model_size:.2f} GB")
    print(f"Delta model size: {delta_model_size:.2f} GB")
