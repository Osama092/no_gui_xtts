import os
import sys
import tempfile
from pathlib import Path
import shutil
import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list, find_latest_best_model, list_audios
from utils.gpt_train import train_gpt

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Global variables - Hardcoded settings
VERSION = "v2.0.2"
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACUMM = 1
MAX_AUDIO_LENGTH = 20  # in seconds

# Get the base directory
BASE_DIR = Path(__file__).parent
OUTPUT_PATH = BASE_DIR / "finetune_models"
TRAIN_CSV = OUTPUT_PATH / "dataset" / "metadata_train.csv"
EVAL_CSV = OUTPUT_PATH / "dataset" / "metadata_eval.csv"
LANG_FILE_PATH = OUTPUT_PATH / "dataset" / "lang.txt"

# Global model variable
XTTS_MODEL = None

# Clear logs
def remove_log_file(file_path):
    log_file = Path(file_path)
    if log_file.exists() and log_file.is_file():
        log_file.unlink()

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to run the previous steps or manually set the model paths!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty, repetition_penalty, top_k, top_p, sentence_split, use_config):
    if XTTS_MODEL is None or not speaker_audio_file:
        return "You need to run the previous step to load the model!", None, None

    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    
    if use_config:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=XTTS_MODEL.config.temperature,
            length_penalty=XTTS_MODEL.config.length_penalty,
            repetition_penalty=XTTS_MODEL.config.repetition_penalty,
            top_k=XTTS_MODEL.config.top_k,
            top_p=XTTS_MODEL.config.top_p,
            enable_text_splitting=True
        )
    else:
        out = XTTS_MODEL.inference(
            text=tts_text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=float(repetition_penalty),
            top_k=top_k,
            top_p=top_p,
            enable_text_splitting=sentence_split
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "Speech generated!", out_path, speaker_audio_file

def load_params_tts(out_path, version):
    out_path = Path(out_path)
    ready_model_path = out_path / "ready" 

    vocab_path = ready_model_path / "vocab.json"
    config_path = ready_model_path / "config.json"
    speaker_path = ready_model_path / "speakers_xtts.pth"
    reference_path = ready_model_path / "reference.wav"

    model_path = ready_model_path / "model.pth"

    if not model_path.exists():
        model_path = ready_model_path / "unoptimize_model.pth"
        if not model_path.exists():
            return "Params for TTS not found", "", "", ""

    return "Params for TTS loaded", model_path, config_path, vocab_path, speaker_path, reference_path
    
def train_model():
    # Use hardcoded values
    clear_gpu_cache()

    # Set custom_model to empty string instead of None
    custom_model = ""  # Empty string instead of None
    
    # Convert output path to Path object if it's a string
    output_path = Path(OUTPUT_PATH)
    run_dir = output_path / "run"

    # Remove train dir if exists
    if run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except PermissionError:
            print(f"Cannot remove {run_dir} due to permission error")

    # Read language from file
    language = "en"  # Default
    if LANG_FILE_PATH.exists():
        with open(LANG_FILE_PATH, 'r', encoding='utf-8') as lang_file:
            language = lang_file.read().strip()

    # Validate train and eval CSV files
    if not TRAIN_CSV.exists() or not EVAL_CSV.exists():
        return "Training or evaluation CSV files not found!", "", "", "", ""

    try:
        # Convert seconds to waveform frames
        max_audio_length = int(MAX_AUDIO_LENGTH * 22050)
        
        speaker_xtts_path, config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
            custom_model, 
            VERSION, 
            language, 
            NUM_EPOCHS, 
            BATCH_SIZE, 
            GRAD_ACUMM, 
            str(TRAIN_CSV), 
            str(EVAL_CSV), 
            output_path=str(output_path), 
            max_audio_length=max_audio_length
        )
    except Exception:
        error = traceback.format_exc()
        return f"Training was interrupted due to an error! Please check the console.\nError summary: {error}", "", "", "", ""

    # Set up directories for the model
    ready_dir = output_path / "ready"
    
    # Ensure the ready directory exists
    ready_dir.mkdir(exist_ok=True, parents=True)

    # Copy the best model to the ready directory
    ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
    shutil.copy(ft_xtts_checkpoint, ready_dir / "unoptimize_model.pth")

    # Move reference audio to output folder
    speaker_reference_path = Path(speaker_wav)
    speaker_reference_new_path = ready_dir / "reference.wav"
    shutil.copy(speaker_reference_path, speaker_reference_new_path)

    print("Model training done!")
    return "Model training done!"

def optimize_model():
    out_path = OUTPUT_PATH
    clear_train_data = "run"  # Default to clearing only run directory
    
    out_path = Path(out_path)
    ready_dir = out_path / "ready"
    run_dir = out_path / "run"
    dataset_dir = out_path / "dataset"

    # Clear specified training data directories
    if clear_train_data in {"run", "all"} and run_dir.exists():
        try:
            shutil.rmtree(run_dir)
        except PermissionError as e:
            print(f"An error occurred while deleting {run_dir}: {e}")

    if clear_train_data in {"dataset", "all"} and dataset_dir.exists():
        try:
            shutil.rmtree(dataset_dir)
        except PermissionError as e:
            print(f"An error occurred while deleting {dataset_dir}: {e}")

    # Get full path to model
    model_path = ready_dir / "unoptimize_model.pth"

    if not model_path.is_file():
        return "Unoptimized model not found in ready folder", ""

    # Load the checkpoint and remove unnecessary parts
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    del checkpoint["optimizer"]

    for key in list(checkpoint["model"].keys()):
        if "dvae" in key:
            del checkpoint["model"][key]

    # Remove unoptimized model
    os.remove(model_path)

    # Save the optimized model
    optimized_model_file_name = "model.pth"
    optimized_model = ready_dir / optimized_model_file_name
    torch.save(checkpoint, optimized_model)

    clear_gpu_cache()
    return "Model optimized and saved"

if __name__ == "__main__":
    # Choose the operation - Default to training
    # You can change this variable to "optimize" if you want to optimize instead
    operation = "train"  # or "optimize"
    
    if operation == "optimize":
        result = optimize_model()
    else:
        # No longer passing custom_model parameter
        result = train_model()
    
    print(result)