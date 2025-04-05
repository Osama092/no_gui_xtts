Trainined audio = 2min
epochs = 1
whisper model = large v3

(envv) C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui>python quant.py
Type of loaded file: <class 'dict'>
Top-level keys: ['config', 'model', 'scaler', 'step', 'epoch', 'date', 'model_loss']
Model key exists
Number of parameters in model: 965
Sample keys: ['xtts.mel_stats', 'xtts.hifigan_decoder.waveform_decoder.conv_pre.bias', 'xtts.hifigan_decoder.waveform_decoder.conv_pre.weight', 'xtts.hifigan_decoder.waveform_decoder.ups.0.bias', 'xtts.hifigan_decoder.waveform_decoder.ups.0.parametrizations.weight.original0']
Type of config: <class 'dict'>
Type of model: <class 'collections.OrderedDict'>
Type of scaler: <class 'NoneType'>
Type of step: <class 'int'>
Type of epoch: <class 'int'>
Quantized state dict saved to C:\Users\oussa\OneDrive\Desktop\Folder\Maybe\Audio cloning stuff\xtts-finetune-webui\finetune_models\ready\quantized_model_state_dict.pth

# Raw
## Load Time - 3 tries

### Base model
Model took 15.48 seconds to complete.  
Model took 13.27 seconds to complete.  
Model took 13.77 seconds to complete.

### Optimized model
separte load:
Model took 14.71 seconds to complete.  
Model took 13.75 seconds to complete.  
Model took 14.42 seconds to complete.

load in app:
Model took 25.43 seconds to complete.
Model took 25.70 seconds to complete.
Model took 25.77 seconds to complete.
### Inference - 3 tries - reference wav 11 sec - input text "This model sounds really good and above all, it's reasonably fast." - advacneced setting: defualt

Inference took 26.65 seconds to complete.
Inference took 26.03 seconds to complete.
Inference took 24.11 seconds to complete.
# Quanting

## Load Time
Model took 24.57 seconds to complete.
Model took 23.49 seconds to complete.
Model took 24.10 seconds to complete.

## Inference Time
Inference took 18.82 seconds to complete.
Inference took 18.99 seconds to complete.
Inference took 18.64 seconds to complete.

# Pruning
## Before
Total Weight Parameters: 465347488
Zero Weight Parameters: 2
Sparsity: 0.00%


## After
Total Weight Parameters: 465347488
Zero Weight Parameters: 24131866
Sparsity: 5.19%

### Loading
Model took 24.79 seconds to complete.
Model took 24.02 seconds to complete.
Model took 25.60 seconds to complete.
### Inference
Model took 65.17 seconds to complete.
Model took 68.15 seconds to complete.
Model took 60.45 seconds to complete.