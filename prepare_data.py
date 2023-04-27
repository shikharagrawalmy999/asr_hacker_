import os
import librosa
from tqdm import tqdm
import torch
import numpy as np
import codecs
from utils.text2seq import text2seq
from layers import TacotronSTFT
import hparams as hp

csv_file = './data/metadata.csv'
root_dir = './data/wavs'
data_dir = './data/preprocessed'

stft = TacotronSTFT()
def get_mel(filename):
    wav, sr = librosa.load(filename, sr=hp.sampling_rate)
    wav = torch.FloatTensor(wav.astype(np.float32))
    
    ### trimming ###
    start = torch.where(torch.abs(wav)>(torch.abs(wav).max()*0.05))[0][0]
    end = torch.where(torch.abs(wav)>(torch.abs(wav).max()*0.05))[0][-1]
    
    ### 50ms silence padding ###
    wav = torch.nn.functional.pad(wav[start:end], (0, hp.sampling_rate//20))
    melspec = stft.mel_spectrogram(wav.unsqueeze(0))
    
    return melspec.squeeze(0), wav


if not os.path.exists(f'{data_dir}'):
    os.mkdir(f'{data_dir}')
if not os.path.exists(f'{data_dir}/phone_seq'):
    os.mkdir(f'{data_dir}/phone_seq')
if not os.path.exists(f'{data_dir}/melspectrogram'):
    os.mkdir(f'{data_dir}/melspectrogram')


with codecs.open(csv_file, 'r', 'utf-8') as f:
    for line in tqdm(f.readlines()):
        fname, _, text = line.split("|")
        wav_name = os.path.join(root_dir, fname) + '.wav'
        phone_seq = text2seq(text)
        melspec, wav = get_mel(wav_name)
        np.save(f'{data_dir}/phone_seq/{fname}_sequence.npy', phone_seq)
        np.save(f'{data_dir}/melspectrogram/{fname}_melspectrogram.npy', melspec.numpy())
    
print("FINISH DATA PREPROCESSING!!!")