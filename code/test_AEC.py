########################################################
# Imports
########################################################

import tensorflow as tf
import random
import numpy as np
import pandas as pd
from utils import *
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

# GPU selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


random.seed(42)
np.random.seed(0)

########################################################
# Data directories and folders
########################################################

dir_dataset = '../datasets/synthetic/'
dir_results = '../results/'
with open(dir_results + 'normalization_values.npy') as f:
    normalization = json.load(f)
if not os.path.isdir(dir_results + 'predictions_cnnMOS/'):
    os.mkdir(dir_results + 'predictions_cnnMOS/')

########################################################
# PARAMERES
########################################################

method = 'baseline'

'Constants'
# AEC processing characteristics
Tprev = 140
Tnow = 10
Tpost = 10
NFFT = 320  # fft points

########################################################
# Main
########################################################

# Get model
model, criterion, optimizer = u_net_2d(input_size=(160, 32), optimizer='nadam', learning_rate=0.01, mode=2,
                                       number_filters_0=16, resize_factor_0=[1, 1])
model.load_weights('model_last')


if 'synthetic' in dir_dataset:

    # Select number files from dataset
    files = os.listdir(dir_dataset + 'nearend_mic_signal')[0:400]

    # Pre-allocate fetures
    nRecordings = len(files)

    # Load signals and process features
    for iRecording in np.arange(0, nRecordings):
        print(str(iRecording+1) + '/' + str(nRecordings), end='\r')

        fs, fes = readwav(dir_dataset + 'farend_speech' + '/' + 'farend_speech_fileid_' + str(iRecording) + '.wav')  # fes: far-end signal
        fs, nem = readwav(dir_dataset + 'nearend_mic_signal' + '/' + 'nearend_mic_fileid_' + str(iRecording) + '.wav')  # nem: near-end microphone
        fs, nes = readwav(dir_dataset + 'nearend_speech' + '/' + 'nearend_speech_fileid_' + str(iRecording) + '.wav')  # near-end speech

        s = np.zeros(nem.shape)
        n0 = int((Tprev + Tnow) * 1e-3*fs)
        while n0 + int((Tpost* 1e-3) * fs) < fes.shape[0]:
            # Signal windowing
            fes_i = fes[n0-int((Tprev+Tnow) * 1e-3 * fs): n0 + int((Tpost * 1e-3) * fs)].astype('float16')
            nem_i = nem[n0-int((Tprev+Tnow) * 1e-3 * fs): n0 + int((Tpost * 1e-3) * fs)].astype('float16')
            nes_i = nes[n0-int((Tprev+Tnow) * 1e-3 * fs): n0 + int((Tpost * 1e-3) * fs)].astype('float16')

            # Feature computation
            stft_fes_i = (librosa.core.stft(fes_i, n_fft=NFFT, hop_length=int(NFFT * 0.25), win_length=320, window='hann', center=True))[0:-1, 1:]
            stft_nem_i = (librosa.core.stft(nem_i, n_fft=NFFT, hop_length=int(NFFT * 0.25), win_length=320, window='hann', center=True))[0:-1, 1:]

            stft_nem_i_phase = np.angle(stft_nem_i)

            stft_fes_i = norm_spectrogram(stft_fes_i, 320, fs, mel=True)
            stft_nem_i = norm_spectrogram(stft_nem_i, 320, fs, mel=True)

            # Prediction
            pred = model.predict(np.concatenate([np.expand_dims(stft_nem_i, (0, -1)), np.expand_dims(stft_fes_i, (0, -1))], axis=-1))
            pred = np.squeeze(pred)

            pred = stft_to_mel(pred, fs)
            pred = mel_to_stft(pred, fs)

            s_i = recover_from_spectrogram(pred, stft_nem_i_phase, fs, mel=True)

            # ISTFT
            s[n0-int((Tnow*1e-3) * fs):n0] = s_i[int(Tprev * 1e-3 * fs):int((Tprev+Tnow)*1e-3 * fs)]

            n0 += int((Tpost* 1e-3) * fs)

        writewav(dir_results + 'predictions_cnnMOS/' + str(iRecording) + '.wav', fs, s)
