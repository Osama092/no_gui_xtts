import os
import sys
import tempfile
from pathlib import Path
import time
import shutil
import glob

import gradio as gr
import librosa.display
import numpy as np

import torch
import torchaudio
import traceback
from utils.formatter import format_audio_list, find_latest_best_model, list_audios
from utils.gpt_train import train_gpt

from faster_whisper import WhisperModel

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Hardcoded parameters (previously command line arguments)
OUT_PATH = str(Path.cwd() / "finetune_models")
SPEAKER_REFERENCE_AUDIO = str(Path.cwd() / "finetune_models" / "ready" / "reference.wav")
TTS_TEXT = "This model sounds really good and above all, it's reasonably fast."
XTTS_CHECKPOINT = str(Path.cwd() / "finetune_models" / "pruned" / "model_pruned.pth")
XTTS_CONFIG = str(Path.cwd() / "finetune_models" / "ready" / "config.json")
XTTS_VOCAB = str(Path.cwd() / "finetune_models" / "ready" / "vocab.json")
XTTS_SPEAKER = str(Path.cwd() / "finetune_models" / "ready" / "speakers_xtts.pth")
VERSION = "v2.0.2"
TEMPERATURE = 0.75
LENGTH_PENALTY = 1
REPETITION_PENALTY = 5
TOP_K = 50
TOP_P = 0
SENTENCE_SPLIT = False
USE_CONFIG = False
LANGUAGE = "en"

# Clear logs
def remove_log_file(file_path):
     log_file = Path(file_path)
     if log_file.exists() and log_file.is_file():
         log_file.unlink()

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None

def load_model(xtts_checkpoint, xtts_config, xtts_vocab, xtts_speaker):
    start_time = time.time()

    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        return "You need to set the XTTS checkpoint, config, and vocab paths properly!"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")


    end_time = time.time()
    duration = end_time - start_time
    print(f"Model took {duration:.2f} seconds to complete.")
    return "Model Loaded!"

def run_tts(lang, tts_text, speaker_audio_file, temperature, length_penalty, repetition_penalty, top_k, top_p, sentence_split, use_config):
    start_time = time.time()

    if XTTS_MODEL is None or not speaker_audio_file:
        return "Model not loaded or speaker audio file not specified!", None, None

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

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"Speech generated successfully! File saved at: {out_path}")
    else:
        print("Failed to generate speech or the file is empty.")

    end_time = time.time()
    duration = end_time - start_time

    print(f"Model took {duration:.2f} seconds to complete.")

    return "Speech generated!", out_path, speaker_audio_file


if __name__ == "__main__":
    # Load the XTTS model using hardcoded parameters
    result = load_model(
        xtts_checkpoint=XTTS_CHECKPOINT,
        xtts_config=XTTS_CONFIG,
        xtts_vocab=XTTS_VOCAB,
        xtts_speaker=XTTS_SPEAKER
    )
    print(result)  # Print the model loading result

    # Ensure the model was loaded successfully
    if XTTS_MODEL is None:
        print("Model loading failed. Exiting.")
        sys.exit(1)

    # Call the run_tts function with hardcoded parameters
    result, out_path, speaker_audio_file = run_tts(
        lang=LANGUAGE,
        tts_text=TTS_TEXT,
        speaker_audio_file=SPEAKER_REFERENCE_AUDIO,
        temperature=TEMPERATURE,
        length_penalty=LENGTH_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
        top_k=TOP_K,
        top_p=TOP_P,
        sentence_split=SENTENCE_SPLIT,
        use_config=USE_CONFIG
    )

    print(result)
    if out_path:
        print(f"Output file: {out_path}")
    if speaker_audio_file:
        print(f"Speaker audio file: {speaker_audio_file}")