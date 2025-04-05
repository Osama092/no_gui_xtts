import os
import sys
import tempfile
from pathlib import Path
import shutil
import glob
import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list, find_latest_best_model, list_audios
from utils.gpt_train import train_gpt
from faster_whisper import WhisperModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def remove_log_file(file_path):
    log_file = Path(file_path)
    if log_file.exists() and log_file.is_file():
        log_file.unlink()

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None

def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to set the XTTS checkpoint, config, and vocab paths!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model!")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()
    print("Model Loaded!")
    return "Model Loaded!"

def preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path):
    print(f"Audio Path: {audio_path}")
    print(f"Output Path: {out_path}")
    print(f"Language: {language}")
    print(f"Whisper Model: {whisper_model}")

    clear_gpu_cache()
    out_path = os.path.join(out_path, "dataset")
    os.makedirs(out_path, exist_ok=True)

    if audio_folder_path:
        audio_files = list(list_audios(audio_folder_path))
    else:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_files = [audio_path]

    if not audio_files:
        return "No audio files found!"
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        asr_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
        train_meta, eval_meta, audio_total_size = format_audio_list(audio_files, asr_model=asr_model, target_language=language, out_path=out_path)
    except Exception as e:
        traceback.print_exc()
        return f"Data processing error: {e}"

    if audio_total_size < 120:
        return "Audio duration must be at least 2 minutes!"

    print("Dataset Processed!")
    return "Dataset Processed!"

if __name__ == "__main__":
    out_path = os.path.join(os.getcwd(), "finetune_models")
    audio_path = "C:\\Users\\oussa\Downloads\\3min_training_audio.wav"
    audio_folder_path = ""
    language = "en"
    whisper_model = "large-v3"

    preprocess_dataset(audio_path, audio_folder_path, language, whisper_model, out_path)
