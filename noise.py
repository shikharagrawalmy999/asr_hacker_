import librosa

import matplotlib.pyplot as plt
import numpy as np
import math

file_path = "./test/LJ045-0096.wav"
signal, sr = librosa.load(file_path, sr = 22050)
signalc, src = librosa.load("./test/test_0.wav", sr = 22050)
RMS=math.sqrt(np.mean(signal**2))

STD_n= 0.1
noise=np.random.normal(0, STD_n, signal.shape[0])
signal_noise = 0.5*signal[:53248]+1.2*signalc
import soundfile
soundfile.write('./test/LJ045-0096_ours.wav',signal_noise,22050)