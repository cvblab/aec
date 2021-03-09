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

dir_dataset = '../datasets/blind_test_set/'
dir_results = '../results/unets_minMaxNorm_fullDB_tanh/'
if not os.path.isdir(dir_results + 'predictions_blind_test/'):
    os.mkdir(dir_results + 'predictions_blind_test/')

########################################################
# PARAMETERS
########################################################

'Constants'
# AEC processing characteristics
Tprev = 130
Tnow = 20
Tpost = 10
NFFT = 320  # fft points

########################################################
# Main
########################################################

# Get model
model, criterion, optimizer = u_net_2d(input_size=(160, 32), optimizer='sgd', learning_rate=1, mode=2,
                                           number_filters_0=16, resize_factor_0=[1, 1], bn=True)
model.load_weights(dir_results + 'model_best')

# Select number files from dataset
directories = [dir_dataset + 'clean/', dir_dataset + 'noisy/']
files = []
clean = []
for iDirectory in directories:

    # Select number files from dataset
    iFiles = os.listdir(iDirectory)
    iFiles = [iDirectory + '_'.join(iiFile.split('_')[:-1]) for iiFile in iFiles if 'lpb' in iiFile]

    if 'clean' in iDirectory:
        files.extend(iFiles)
        clean.extend(list(np.ones(len(iFiles))))
    else:
        files.extend(iFiles)
        clean.extend(list(np.zeros(len(iFiles))))

nRecordings = len(files)
for iRecording in np.arange(0, nRecordings):
    print(str(iRecording) + '/' + str(nRecordings), end='\r')

    if os.path.isfile(dir_results + 'predictions_blind_test/' + '/' + files[iRecording].split('/')[-1] + '.wav'):
        continue

    if 'clean' in files[iRecording]:
        fs, nem = readwav(files[iRecording] + '_mic.wav')
        #fs, nem = readwav(files[iRecording] + '_mic_c.wav')
    else:
        fs, nem = readwav(files[iRecording] + '_mic.wav')
    fs, fes = readwav(files[iRecording] + '_lpb.wav')

    if (nem.shape[0] - fes.shape[0]) < 0:
        fes = fes[-(nem.shape[0] - fes.shape[0]):]
    elif (nem.shape[0] - fes.shape[0]) > 0:
        nem = nem[(nem.shape[0] - fes.shape[0]):]

    stft_nem = np.abs(
        librosa.core.stft(nem, n_fft=NFFT, hop_length=int(NFFT * 0.25), win_length=320, window='hann', center=True))

    s = np.zeros(nem.shape)
    n0 = int((Tprev + Tnow) * 1e-3 * fs)
    while n0 + int((Tnow * 1e-3) * fs) < fes.shape[0]:
        t0 = timer()
        # Signal windowing
        fes_i = fes[n0 - int((Tprev + Tnow) * 1e-3 * fs): n0 + int((Tpost * 1e-3) * fs)].astype('float16')
        nem_i = nem[n0 - int((Tprev + Tnow) * 1e-3 * fs): n0 + int((Tpost * 1e-3) * fs)].astype('float16')

        # Feature computation
        stft_fes_i = (librosa.core.stft(fes_i, n_fft=NFFT, hop_length=int(NFFT * 0.25), win_length=320, window='hann',
                                        center=True))[:, 1:]
        stft_nem_i = (librosa.core.stft(nem_i, n_fft=NFFT, hop_length=int(NFFT * 0.25), win_length=320, window='hann',
                                        center=True))[:, 1:]

        stft_nem_i_phase = np.angle(stft_nem_i)
        stft_nem_i = stft_nem_i[:-1, :]
        stft_fes_i = stft_fes_i[:-1, :]

        stft_fes_i = 2. * (np.abs(stft_fes_i) - np.min(stft_nem)) / (np.ptp(stft_nem) + 1e-6) - 1
        stft_nem_i = 2. * (np.abs(stft_nem_i) - np.min(stft_nem)) / (np.ptp(stft_nem) + 1e-6) - 1

        # Prediction
        t0_net = timer()
        pred = model.predict(
            np.concatenate([np.expand_dims(stft_nem_i, (0, -1)), np.expand_dims(stft_fes_i, (0, -1))], axis=-1))
        pred = np.squeeze(pred)
        et_net = timer() - t0_net

        # Recover odd dimension

        # stft_fes_i = (stft_fes_i * (np.std(stft_nem) + 1e-6)) + np.mean(stft_nem)
        pred = ((pred + 1) * (np.ptp(stft_nem) + 1e-6) / 2) + np.min(stft_nem)
        pred = np.concatenate([pred, np.zeros((1, pred.shape[1]))])
        pred = pred.astype(np.complex) * np.exp(1j * stft_nem_i_phase)

        s_i = librosa.istft(pred, hop_length=int(NFFT * 0.25), win_length=320, window='hann', center=True)

        # ISTFT
        if Tpost == 0:
            s[n0 - int((Tnow * 1e-3) * fs):n0] = s_i[-int((Tnow + Tpost) * 1e-3 * fs):]
        else:
            s[n0 - int((Tnow * 1e-3) * fs):n0] = s_i[-int((Tnow + Tpost) * 1e-3 * fs):-int((Tpost) * 1e-3 * fs) ]

        n0 += int((Tnow * 1e-3) * fs)
        et = timer() - t0

        print('et (all): ' + str(et * 1e3) + ' ms, et(aec): ' + str(et_net * 1e3) + ' ms', end='\r')

    writewav(dir_results + 'predictions_blind_test/' + '/' + files[iRecording].split('/')[-1] + '.wav', fs, s)
