import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")
from scipy.io.wavfile import write
import sys
sys.path.append('waveglow/')

import matplotlib.pyplot as plt
# %matplotlib inline

import IPython.display as ipd
from text import *
import torch
import hparams as hp
from modules.model import Model
from denoiser import Denoiser
from utils.utils import *
from utils.text2seq import text2seq

# Supply path to the pretrained waveglow model
# waveglow_path = 'training_log/waveglow_256channels_ljs_v2.pt'

# waveglow = torch.load(waveglow_path, map_location=torch.device("cpu"))['model']
waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32', map_location=torch.device('cpu'))
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cpu')
waveglow.eval()


for m in waveglow.modules():
    if 'Conv' in str(type(m)):
        setattr(m, 'padding_mode', 'zeros')

waveglow.eval()
for k in waveglow.convinv:
    k.float()

denoiser = Denoiser(waveglow)

checkpoint_path = f"training_log/baseline/checkpoint_11000"
model = Model(hp)
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu"))['state_dict'])
_ = model.eval()

with open('filelists/test_new.txt', 'r') as f:
    test_data = f.read().splitlines()
    
for i, x in enumerate(test_data[:10]):
    file, text = x.split('|')
    print(f"{file}: {text}")
    phone_seq = text2seq(text)
    sequence = torch.autograd.Variable(torch.from_numpy(phone_seq)).long().unsqueeze(0)

    temperature=[0.333, 0.333, 0.333, 0.333]
    with torch.no_grad():
        melspec, durations = model.inference(sequence, alpha=1.0, temperature=temperature)
        melspec = melspec*(hp.max_db-hp.min_db)+hp.min_db
        # audio = waveglow.infer(melspec, sigma=0.666)
        audio = waveglow.infer(melspec)
        audio_numpy = audio[0].data.cpu().numpy()
        write(f"test_audio/test_{i}.wav", 22050, audio_numpy)
        # audio_denoised = denoiser(audio, strength=0.03)[:, 0]
        # with open(f"test_audio/test_{i}.wav", "wb") as f:
        #     f.write((ipd.Audio(audio_denoised.cpu().numpy(), rate=22050)).data)
