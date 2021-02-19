# UTILS
########################################################
# Imports
########################################################

import numpy as np
import random
import os
from matplotlib import pyplot as plt
import librosa
from scipy.io import wavfile
from timeit import default_timer as timer
import pandas as pd
import datetime
import json
import tensorflow as tf

########################################################
# Data Generators
########################################################


class Dataset(object):

    def __init__(self, dir_dataset, partition='train', dir_results='../results/', normalization=False, debugging=False):

        'Initialization'
        self.dir_dataset = dir_dataset
        self.partition = partition
        self.dir_results = dir_results

        'Constants'
        self.w = 160
        self.T = 160  # Frame size (ms)
        self.Ts = self.T/2
        self.threshold = 20  # Limit for VAD
        self.NFFT = 320  # fft points

        if 'synthetic' in dir_dataset:
            self.D = 10  # Recordings duration

            # Select number files from dataset
            files = os.listdir(self.dir_dataset + 'nearend_mic_signal')

            if debugging:
                files = files[0:1000]

            if self.partition == 'train':
                files = files[400:]
            elif self.partition == 'test':
                files = files[0:400]

            # Pre-allocate fetures
            nRecordings = len(files)
            nSamples = int(nRecordings*(self.D / (self.w*1e-3)))

            self.NES = np.zeros((nSamples, int(self.NFFT/2), 32), dtype=np.float16)
            self.NEM = np.zeros((nSamples, int(self.NFFT/2), 32), dtype=np.float16)
            self.FES = np.zeros((nSamples, int(self.NFFT/2), 32), dtype=np.float16)

            self.indexes_st = []
            self.indexes_dt = []

            # Load signals and process features
            print('[INFO]: Training on ram: Loading images')
            iSample = 0
            for iRecording in np.arange(0, nRecordings):
                print(str(iSample) + '/' + str(nSamples), end='\r')

                fs, fes = readwav(self.dir_dataset + 'farend_speech' + '/' + 'farend_speech_fileid_' + str(iRecording) + '.wav')
                fs, nem = readwav(self.dir_dataset + 'nearend_mic_signal' + '/' + 'nearend_mic_fileid_' + str(iRecording) + '.wav')
                fs, nes = readwav(self.dir_dataset + 'nearend_speech' + '/' + 'nearend_speech_fileid_' + str(iRecording) + '.wav')

                n0 = 0
                while n0 + int((self.w*1e-3)*fs) < fes.shape[0]:

                    # Signal windowing
                    fes_i = fes[n0:n0 + int((self.w*1e-3)*fs)].astype('float16')
                    nem_i = nem[n0:n0 + int((self.w * 1e-3) * fs)].astype('float16')
                    nes_i = nes[n0:n0 + int((self.w * 1e-3) * fs)].astype('float16')

                    # Feature computation
                    stft_fes_i = np.abs(librosa.core.stft(fes_i, n_fft=self.NFFT, hop_length=int(self.NFFT*0.25),
                                                          win_length=320, window='hann', center=True))
                    stft_nem_i = np.abs(librosa.core.stft(nem_i, n_fft=self.NFFT, hop_length=int(self.NFFT*0.25),
                                                          win_length=320, window='hann', center=True))
                    stft_nes_i = np.abs(librosa.core.stft(nes_i, n_fft=self.NFFT, hop_length=int(self.NFFT*0.25),
                                                          win_length=320, window='hann', center=True))

                    # Determine whether it is double talk or not
                    is_dt, E_nes_i, E_fes_i = double_talk_detector(stft_nes_i, stft_fes_i, self.NFFT, fs,
                                                                   threshold=self.threshold)

                    if is_dt:
                        self.indexes_dt.append(iSample)
                    else:
                        self.indexes_st.append(iSample)

                    stft_fes_i = norm_spectrogram(stft_fes_i, 320, fs, mel=True)
                    stft_nem_i = norm_spectrogram(stft_nem_i, 320, fs, mel=True)
                    stft_nes_i = norm_spectrogram(stft_nes_i, 320, fs, mel=True)

                    self.NES[iSample, :, :] = stft_nes_i[0:-1, 1:]
                    self.NEM[iSample, :, :] = stft_nem_i[0:-1, 1:]
                    self.FES[iSample, :, :] = stft_fes_i[0:-1, 1:]

                    n0 += int((self.w*1e-3)*fs)
                    iSample += 1

            # Keep only added samples
            self.nSamples = iSample
            self.NES = self.NES[:iSample]
            self.NEM = self.NEM[:iSample]
            self.FES = self.FES[:iSample]

        elif 'test_set' in dir_dataset:
            self.D = 15  # Recordings duration
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

            if self.partition == 'train':
                files = files[20:-20]
                clean = clean[20:-20]
            else:
                files = files[:20] + files[-20:]
                clean = clean[:20] + clean[-20:]

            # Pre-allocate fetures
            nRecordings = len(files)
            nSamples = int(nRecordings*(self.D / (self.w*1e-3)))

            self.FES = np.zeros((nSamples, int(self.NFFT/2), 32), dtype=np.float16)
            self.NEM = np.zeros((nSamples, int(self.NFFT/2), 32), dtype=np.float16)
            self.Y = np.zeros(nSamples, dtype=np.float16)

            # Load signals and process features
            print('[INFO]: Training on ram: Loading images')
            iSample = 0
            for iRecording in np.arange(0, nRecordings):
                print(str(iSample) + '/' + str(nSamples), end='\r')

                if 'clean' in files[iRecording]:
                    fs, nem = readwav(files[iRecording] + '_mic_c.wav')
                else:
                    fs, nem = readwav(files[iRecording] + '_mic.wav')
                fs, fes = readwav(files[iRecording] + '_lpb.wav')

                n0 = 0
                while n0 + int((self.w*1e-3)*fs) < fes.shape[0] and n0 + int((self.w*1e-3)*fs) < nem.shape[0]:

                    # Signal windowing
                    fes_i = fes[n0:n0 + int((self.w*1e-3)*fs)].astype('float16')
                    nem_i = nem[n0:n0 + int((self.w * 1e-3) * fs)].astype('float16')

                    stft_fes_i = np.abs(librosa.core.stft(fes_i, n_fft=self.NFFT, hop_length=int(self.NFFT*0.25),
                                                          win_length=320, window='hann', center=True))
                    stft_nem_i = np.abs(librosa.core.stft(nem_i, n_fft=self.NFFT, hop_length=int(self.NFFT*0.25),
                                                          win_length=320, window='hann', center=True))

                    stft_fes_i = norm_spectrogram(stft_fes_i, 320, fs, mel=True)
                    stft_nem_i = norm_spectrogram(stft_nem_i, 320, fs, mel=True)

                    self.NEM[iSample, :, :] = stft_nem_i[0:-1, 1:]
                    self.FES[iSample, :, :] = stft_fes_i[0:-1, 1:]
                    self.Y[iSample] = clean[iRecording]

                    n0 += int((self.w*1e-3)*fs)
                    iSample += 1

            # Keep only added samples
            self.nSamples = iSample
            self.NEM = self.NEM[:iSample]
            self.FES = self.FES[:iSample]
            self.Y = self.Y[:iSample]

            self.indexes = np.arange(0, iSample)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes_dt),len(self.indexes_st)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        idx = index

        # Get features from dataset
        if 'synthetic' in self.dir_dataset:
            nes = self.NES[idx, :, :]
            nem = self.NEM[idx, :, :]
            fes = self.FES[idx, :, :]

            return nes, nem, fes

        elif 'test' in self.dir_dataset:
            nem = self.NEM[idx, :, :]
            fes = self.FES[idx, :, :]
            y = self.Y[idx]

            return y, nem, fes


class DataGenerator(object):

    def __init__(self, dataset, batch_size=32, shuffle=False):

        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        'Secondary Initializations'
        self._idx_st = 0
        self._idx_dt = 0
        self.index_st = dataset.indexes_st
        self.index_dt = dataset.indexes_dt
        self.finished_st = False
        self.finished_dt = False
        self._reset()

    def __len__(self):

        N = np.maximum(len(self.index_dt), len(self.index_st))
        b = self.batch_size/2
        return int(N // b + bool(N % b))

    def __iter__(self):

        return self

    def __next__(self):

        if self._idx_dt + self.batch_size/2 >= len(self.index_dt):
            random.shuffle(self.index_dt)
            self._idx_dt = 0
            self.finished_dt = True

        if self._idx_st + self.batch_size/2 >= len(self.index_st):
            random.shuffle(self.index_st)
            self._idx_st  = 0
            self.finished_st = True

        if self.finished_st and self.finished_dt:
            self._reset()
            raise StopIteration()

        # Load images and include into the batch (st)
        NES_st, NEM_st, FES_st = [], [], []
        for i in np.arange(self._idx_st, self._idx_st+self.batch_size/2):
            nes, nem, fes = self.dataset.__getitem__(self.index_st[int(i)])
            NES_st.append(nes)
            NEM_st.append(nem)
            FES_st.append(fes)

        NES_st = np.expand_dims(np.array(NES_st), 1)
        NEM_st = np.expand_dims(np.array(NEM_st), 1)
        FES_st = np.expand_dims(np.array(FES_st), 1)

        self._idx_st += self.batch_size/2

        # Load images and include into the batch (dt)
        NES_dt, NEM_dt, FES_dt = [], [], []
        for i in np.arange(self._idx_dt, self._idx_dt + self.batch_size/2):
            nes, nem, fes = self.dataset.__getitem__(self.index_dt[int(i)])
            NES_dt.append(nes)
            NEM_dt.append(nem)
            FES_dt.append(fes)

        NES_dt = np.expand_dims(np.array(NES_dt), 1)
        NEM_dt = np.expand_dims(np.array(NEM_dt), 1)
        FES_dt = np.expand_dims(np.array(FES_dt), 1)

        self._idx_dt += self.batch_size/2

        NES = np.concatenate((NES_st, NES_dt), axis=1)
        NEM = np.concatenate((NEM_st, NEM_dt), axis=1)
        FES = np.concatenate((FES_st, FES_dt), axis=1)

        return np.array(NES).astype('float32'), np.array(NEM).astype('float32'),\
               np.array(FES).astype('float32')

    def _reset(self):

        if self.shuffle:
            random.shuffle(self.index_st)
            random.shuffle(self.index_dt)
        self._idx_st = 0
        self._idx_dt = 0
        self.finished_st = False
        self.finished_dt = False


class DataGeneratorCnnMos(object):

    def __init__(self, dataset, batch_size=32, shuffle=False):

        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        'Secondary Initializations'
        self.indexes = dataset.indexes
        self._idx = 0
        self._reset()

    def __len__(self):

        N = len(self.indexes)
        b = self.batch_size
        return int(N // b + bool(N % b))

    def __iter__(self):

        return self

    def __next__(self):

        if self._idx + self.batch_size >= len(self.indexes):
            self._reset()
            raise StopIteration()

        # Load images and include into the batch (st)
        Y, NEM, FES = [], [], []
        for i in np.arange(self._idx, self._idx+self.batch_size):
            y, nem, fes = self.dataset.__getitem__(self.indexes[int(i)])
            Y.append(y)
            NEM.append(nem)
            FES.append(fes)

        self._idx += self.batch_size

        return np.array(Y).astype('float32'), np.array(NEM).astype('float32'),\
               np.array(FES).astype('float32')

    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0


########################################################
# Architectures
########################################################

def CNN_MOS(input_size=(162, 33), optimizer='sgd', learning_rate=1e-2, mode=0, number_filters_0=64, resize_factor_0=[1, 1], res_factor=[2, 1]):

    def encoding_block(input_layer, pooling_factor, number_filters_0, filters_factor, mode_convolution=1):

        x = tf.keras.layers.AveragePooling2D(pool_size=(pooling_factor[0], pooling_factor[1]))(input_layer)  # pooling
        # x = input_layer
        x = convolutional_block_1(x, n_filters=number_filters_0 * filters_factor,
                                  kernel_size=3)  # dimensionality normalization
        # Feature extraction block
        if mode == 0:
            x = convolutional_block_1(x, n_filters=number_filters_0 * filters_factor)
        elif mode == 1:
            x = convolutional_block_2(x, n_filters=number_filters_0 * filters_factor)
        elif mode == 2:
            x = residual_block_1(x, n_filters=number_filters_0 * filters_factor)
        elif mode == 3:
            x = residual_block_2(x, n_filters=number_filters_0 * filters_factor)

        return x

   # Architecture definition
    inputs = tf.keras.Input((input_size[0], input_size[1], 2))

    # ----- ENCODING -----

    # Block 1, 1024 --> 512, Filters_0 x 1
    encoding_1_out = encoding_block(inputs, resize_factor_0, number_filters_0, 1, mode_convolution=mode)
    # Block 2, 512 --> 256, Filters_0 x 2
    encoding_2_out = encoding_block(encoding_1_out, res_factor, number_filters_0, 2, mode_convolution=mode)
    # Block 3, 256 --> 128, Filters_0 x 4
    encoding_3_out = encoding_block(encoding_2_out, res_factor, number_filters_0, 4, mode_convolution=mode)
    # Block 4, 128 --> 64, Filters_0 x 8
    encoding_4_out = encoding_block(encoding_3_out, res_factor, number_filters_0, 8, mode_convolution=mode)

    x = tf.keras.layers.GlobalAveragePooling2D()(encoding_4_out)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    if 'nadam' in optimizer:
        optimizer = tf.keras.optimizers.Nadam(lr=learning_rate)
    elif 'sgd' in optimizer:
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    elif 'adam' in optimizer:
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    criterion = tf.keras.losses.binary_crossentropy

    return model, criterion, optimizer


def u_net_2d(input_size=(162, 33), optimizer='nadam', learning_rate=1e-5, mode=0, number_filters_0=64, resize_factor_0=[1, 1], res_factor=[2, 1]):

    # Mode = 0 --> sequential
    # Mode = 1 --> sequential 2 conv layers
    # Mode = 2 --> residual 1 layer
    # Mode = 3 --> residual 2 layers
    # Mode = 4 --> blocks multi resolution for feature extraction
    def encoding_block(input_layer, pooling_factor, number_filters_0, filters_factor, mode_convolution=1):

        x = tf.keras.layers.AveragePooling2D(pool_size=(pooling_factor[0], pooling_factor[1]))(input_layer)  # pooling
        #x = input_layer
        x = convolutional_block_1(x, n_filters=number_filters_0*filters_factor, kernel_size=3)  # dimensionality normalization
        # Feature extraction block
        if mode == 0:
            x = convolutional_block_1(x, n_filters=number_filters_0*filters_factor)
        elif mode == 1:
            x = convolutional_block_2(x, n_filters=number_filters_0*filters_factor)
        elif mode == 2:
            x = residual_block_1(x, n_filters=number_filters_0*filters_factor)
        elif mode == 3:
            x = residual_block_2(x, n_filters=number_filters_0*filters_factor)

        return x

    def decoding_block(input_layer, skip_connection_layer, pooling_factor, number_filters_0, filters_factor, mode_convolution=1):

        # Deconvolution
        x = tf.keras.layers.UpSampling2D(size=(pooling_factor[0], pooling_factor[1]))(input_layer)
        #x = input_layer
        x = convolutional_block_1(x, n_filters=number_filters_0*filters_factor, kernel_size=3)
        # Skip connection and number of filters normalization
        x = tf.keras.layers.concatenate([skip_connection_layer, x])
        x = convolutional_block_1(x, n_filters=number_filters_0 * filters_factor, kernel_size=3)
        # Feature extraction block
        if mode == 0:
            x = convolutional_block_1(x, n_filters=number_filters_0*filters_factor)
        elif mode == 1:
            x = convolutional_block_2(x, n_filters=number_filters_0*filters_factor)
        elif mode == 2:
            x = residual_block_1(x, n_filters=number_filters_0*filters_factor)
        elif mode == 3:
            x = residual_block_2(x, n_filters=number_filters_0*filters_factor)

        return x

    # Architecture definition

    inputs = tf.keras.Input((input_size[0], input_size[1], 2))

    # ----- ENCODING -----

    # Block 1, 1024 --> 512, Filters_0 x 1
    encoding_1_out = encoding_block(inputs, resize_factor_0, number_filters_0, 1, mode_convolution=mode)
    # Block 2, 512 --> 256, Filters_0 x 2
    encoding_2_out = encoding_block(encoding_1_out, res_factor, number_filters_0, 2, mode_convolution=mode)
    # Block 3, 256 --> 128, Filters_0 x 4
    encoding_3_out = encoding_block(encoding_2_out, res_factor, number_filters_0, 4, mode_convolution=mode)
    # Block 4, 128 --> 64, Filters_0 x 8
    encoding_4_out = encoding_block(encoding_3_out, res_factor, number_filters_0, 8, mode_convolution=mode)

    # ----- DECODING -----

    # Block 2, 64 --> 128, Filters_0 x 4
    decoding_2_out = decoding_block(encoding_4_out, encoding_3_out, res_factor, number_filters_0, 4, mode_convolution=mode)
    # Block 3, 128 --> 256, Filters_0 x 2
    decoding_3_out = decoding_block(decoding_2_out, encoding_2_out, res_factor, number_filters_0, 2, mode_convolution=mode)
    # Block 4, 256 --> 512, Filters_0 x 1
    decoding_4_out = decoding_block(decoding_3_out, encoding_1_out, res_factor, number_filters_0, 1, mode_convolution=mode)

    # ----- OUTPUT -------

    x = tf.keras.layers.UpSampling2D(size=(resize_factor_0[0], resize_factor_0[1]))(decoding_4_out)
    out = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    if 'nadam' in optimizer:
        optimizer = tf.keras.optimizers.Nadam(lr=learning_rate)
    elif 'sgd' in optimizer:
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    elif 'adam' in optimizer:
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    loss = rmse_coef
    metric = rmse_coef_slicing

    criterion = loss
    #model.compile(optimizer=optimizer, loss=loss, metrics=[metric])  # WORKS

    return model, criterion, optimizer


# Blocks definition
def residual_block_1(input_layer, n_filters):

    x = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Add()([x, input_layer])

    return x


def residual_block_2(input_layer, n_filters):

    x = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same')(x)
    x2 = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Add()([x, x2])

    return x


def convolutional_block_1(input_layer, n_filters, kernel_size=3):

    x = tf.keras.layers.Conv2D(n_filters, kernel_size, activation='relu', padding='same')(input_layer)

    return x


def convolutional_block_2(input_layer, n_filters, stride=3):

    x = tf.keras.layers.Conv2D(n_filters, stride, activation='relu', padding='same')(input_layer)
    x = tf.keras.layers.Conv2D(n_filters, stride, activation='relu', padding='same')(x)

    return x


def mse_coef(y_true, y_pred):

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred))

    return loss


def rmse_coef(y_true, y_pred):

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred)) + 1.0e-12)

    return loss


def rmse_coef_slicing(y_true, y_pred):

    y_true = tf.slice(y_true, [0, 0, 20, 0], [32, 160, 12, 1])
    y_pred = tf.slice(y_pred, [0, 0, 20, 0], [32, 160, 12, 1])

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred)))

    return loss

########################################################
# Training
########################################################


class CNNTrainer:

    def __init__(self, n_epochs, criterion, optimizer, train_on_gpu, device, dir_out, save_best_only=False,
                 lr_exp_decay=False, multi_gpu=False, learning_rate_esch_half=False, class_weights=None,
                 slice=False, channel=False, normalization=None, other_model=None, method='aec'):

        'Initialization'
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_on_gpu = train_on_gpu
        self.save_best_only = save_best_only
        self.device = device
        self.dir_out = dir_out
        self.lr_exp_decay = lr_exp_decay
        self.lr0 = 0 # optimizer.param_groups[0]['lr']
        self.multi_gpu = multi_gpu
        self.learning_rate_esch_half = learning_rate_esch_half
        self.slice = slice
        self.class_weights = class_weights
        self.channel = channel
        self.normalization = normalization
        self.other_model = other_model
        self.method = method # 'aec', 'aec_perceptual'

        'Secondary Initializations'
        self.history = []
        self.start_time_epoch = []
        self.valid_loss_min = float('inf')

        if self.normalization is not None:
            with open(dir_out + 'normalization_values.npy') as f:
                self.normalization = json.load(f)

    def train(self, model, train_generator, val_generator=None, epochs_for_weights_update=1):

        for epoch in range(self.n_epochs):

            print('Epoch: {}'.format(epoch + 1) + '/' + str(self.n_epochs))
            self.start_time_epoch = timer()
            self.epoch = epoch

            # decrease lr after half iterations
            if self.learning_rate_esch_half:
                if epoch == (self.n_epochs / 2) - 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr0 / 10
                    self.lr0 = self.lr0 / 10

            # exponential lr in last epochs
            if self.lr_exp_decay:
                self.optimizer = exp_lr_scheduler(self.optimizer, epoch, lr_decay_epoch=self.n_epochs - 5, lr0=self.lr0)

            # Update model weights
            print('Training...')
            if 'test_set' in train_generator.dataset.dir_dataset:
                metric_train, loss_train = self.train_epoch_cnn_mos(model, train_generator, is_training=True)
            else:
                if self.method == 'aec_perceptual':
                    metric_train, loss_train, mse_train, mos_train = self.train_epoch(model, train_generator, is_training=True)
                elif self.method == 'aec':
                    metric_train, loss_train = self.train_epoch(model, train_generator, is_training=True)

            # Track training progress
            if self.method == 'aec_perceptual':
                print("[INFO TRAINING] Epoch {}/{} : mse={:.6f}, cnn_MOS={:.6f}, overall={:.6f}".format(self.epoch + 1, self.n_epochs, mse, mos, loss_train), end='\r')
            elif self.method == 'aec':
                print("[INFO TRAINING] Epoch {}/{} : mse={:.6f}".format(self.epoch + 1, self.n_epochs, loss_train))

            # Validation performance
            if val_generator is not None:
                print('')
                print('Evaluating validation data...')
                if 'test_set' in train_generator.dataset.dir_dataset:
                    metric_val, loss_val = self.train_epoch_cnn_mos(model, val_generator, is_training=False)
                else:
                    if self.method == 'aec_perceptual':
                        metric_val, loss_val, mse_val, mos_val = self.train_epoch(model, val_generator, is_training=False)
                    elif self.method == 'aec':
                        metric_val, loss_val = self.train_epoch(model, val_generator, is_training=False)
            else:
                metric_val, loss_val, mse, mos = 0, 0, 0, 0

            # Track validation progress
            if self.method == 'aec_perceptual':
                print("[INFO VALIDATION] Epoch {}/{} : mse={:.6f}, cnn_MOS={:.6f}, overall={:.6f}".format(self.epoch + 1, self.n_epochs, mse_val, mos_val, loss_val), end='\r')
            elif self.method == 'aec':
                print("[INFO VALIDATION] Epoch {}/{} : mse={:.6f}".format(self.epoch + 1, self.n_epochs, loss_val))

            # Save best model option
            if self.save_best_only is not False:
                if loss_val < self.valid_loss_min:
                    print('Validation loss improved from ' + str(round(self.valid_loss_min, 5)) + ' to ' + str(
                        round(loss_val, 5)) + '  ... saving model')
                    # Save model
                    if self.multi_gpu is False:
                        model.save_weights(self.dir_out + 'model_best')
                    else:
                        model.save_weights(self.dir_out + 'model_best')
                    # Track improvement
                    self.valid_loss_min = loss_val
                else:
                    print('Validation loss did not improve from ' + str(round(self.valid_loss_min, 5)))
            model.save_weights(self.dir_out + 'model_last')

            # Add metrics for history tracking
            if self.method == 'aec':
                self.history.append([loss_train, loss_val, metric_train, metric_val])
                history_on_epoch = pd.DataFrame(
                    self.history,
                    columns=['loss_train', 'loss_val', 'metric_train', 'metric_val'])

            elif self.method == 'aec_perceptual':
                self.history.append([loss_train, loss_val, metric_train, metric_val, mse_train, mse_val, mos_train, mos_val])
                history_on_epoch = pd.DataFrame(
                    self.history,
                    columns=['loss_train', 'loss_val', 'metric_train', 'metric_val', 'mse_train', 'mse_val', 'mos_train', 'mos_val'])

            # Save history on streaming
            history_on_epoch.to_excel(self.dir_out + 'lc_on_direct.xlsx')

        return model, history_on_epoch

    def train_epoch(self, model, generator, is_training=True, epochs_for_weights_update=1):

        start = timer()

        # keep track of training and validation loss each epoch
        loss_over_all = 0.0
        metric_over_all = 0.0
        mos_over_all = 0.0
        mse_over_all = 0.0

        # Loop over batches in the generator
        for ii, (nes, nem, fes) in enumerate(generator):
            if is_training:  # Training

                with tf.GradientTape() as tape:

                    output_st = model(np.concatenate([np.expand_dims(nem[:, 0, :, :], -1), np.expand_dims(fes[:, 0, :, :], -1)], axis=-1))
                    output_dt = model(np.concatenate([np.expand_dims(nem[:, 1, :, :], -1), np.expand_dims(fes[:, 1, :, :], -1)], axis=-1))

                    alpha = 0.5
                    # Loss and back-propagation of gradients
                    loss = alpha * self.criterion(output_st, np.expand_dims(nes[:, 0, :, :], -1)) + (1 - alpha) * self.criterion(output_dt, np.expand_dims(nes[:, 1, :, :], -1))

                    # Use pre-trained model to incorporate quality perceptual losses
                    if self.method == 'aec_perceptual':
                        # Add cnnMOS
                        L1 = loss
                        fes = np.zeros(nem.shape)

                        quality_st = tf.squeeze(self.other_model(tf.concat([output_st, np.expand_dims(fes[:, 0, :, :], -1)], axis=-1)))
                        quality_dt = tf.squeeze(self.other_model(tf.concat([output_dt, np.expand_dims(fes[:, 1, :, :], -1)], axis=-1)))

                        quality_st_real = tf.squeeze(self.other_model(tf.concat([np.expand_dims(nes[:, 0, :, :], -1), np.expand_dims(fes[:, 0, :, :], -1)], axis=-1)))
                        quality_dt_real = tf.squeeze(self.other_model(tf.concat([np.expand_dims(nes[:, 1, :, :], -1), np.expand_dims(fes[:, 1, :, :], -1)], axis=-1)))

                        L2 = alpha * rmse_coef(quality_st, quality_st_real) + (1 - alpha) * rmse_coef(quality_dt, quality_dt_real)
                        L2 = tf.dtypes.cast(L2, tf.float32)

                        alpha2 = 1e-4
                        loss = L1 + alpha2*L2

                grads = tape.gradient(loss, model.trainable_weights)

                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

            else:  # Validation

                output_st = model(np.concatenate([np.expand_dims(nem[:, 0, :, :], -1), np.expand_dims(fes[:, 0, :, :], -1)], axis=-1))
                output_dt = model(np.concatenate([np.expand_dims(nem[:, 1, :, :], -1), np.expand_dims(fes[:, 1, :, :], -1)], axis=-1))

                alpha = 0.5
                # Loss and back-propagation of gradients
                loss = alpha * self.criterion(output_st, np.expand_dims(nes[:, 0, :, :], -1)) + (1 - alpha) * self.criterion(output_dt, nes[:, 1, :, :])

                # Use pre-trained model to incorporate quality perceptual losses
                if self.method == 'aec_perceptual':
                    # Add cnnMOS
                    L1 = loss
                    fes = np.zeros(nem.shape)

                    quality_st = tf.squeeze(
                        self.other_model(tf.concat([output_st, np.expand_dims(fes[:, 0, :, :], -1)], axis=-1)))
                    quality_dt = tf.squeeze(
                        self.other_model(tf.concat([output_dt, np.expand_dims(fes[:, 1, :, :], -1)], axis=-1)))

                    quality_st_real = tf.squeeze(self.other_model(
                        tf.concat([np.expand_dims(nes[:, 0, :, :], -1), np.expand_dims(fes[:, 0, :, :], -1)], axis=-1)))
                    quality_dt_real = tf.squeeze(self.other_model(
                        tf.concat([np.expand_dims(nes[:, 1, :, :], -1), np.expand_dims(fes[:, 1, :, :], -1)], axis=-1)))

                    L2 = alpha * rmse_coef(quality_st, quality_st_real) + (1 - alpha) * rmse_coef(quality_dt,
                                                                                                  quality_dt_real)
                    L2 = tf.dtypes.cast(L2, tf.float32)

                    alpha2 = 1e-4
                    loss = L1 + alpha2 * L2

            # Track train loss by multiplying average loss by number of examples in batch
            loss_over_all += loss.numpy()
            metric_over_all += loss.numpy()

            if self.method == 'aec_perceptual':

                mse_over_all += L1.numpy()
                mos_over_all += L2.numpy()
                # Track training progress
                print("[INFO] Epoch {}/{} -- Step {}/{}: mse={:.6f}, cnn_MOS={:.6f},"
                      " overall={:.6f}".format(self.epoch + 1, self.n_epochs, ii + 1, len(generator), L1.numpy(), L2.numpy(), loss.numpy()), end='\r')
            else:
                # Track training progress
                print("[INFO] Epoch {}/{} -- Step {}/{}: mse={:.6f} ".format(self.epoch+1, self.n_epochs, ii + 1, len(generator), loss.numpy()), end='\r')

        # Calculate average loss and metrics
        loss_over_all = loss_over_all / len(generator)
        metric_over_all = metric_over_all / len(generator)

        if self.method == 'aec':
            return metric_over_all, loss_over_all
        elif self.method == 'aec_perceptual':
            return metric_over_all, loss_over_all, mse_over_all, mos_over_all

    def train_epoch_cnn_mos(self, model, generator, is_training=True, epochs_for_weights_update=1):

        start = timer()

        # keep track of training and validation loss each epoch
        loss_over_all = 0.0
        metric_over_all = 0.0

        # Loop over batches in the generator
        for ii, (y, nem, fes) in enumerate(generator):
            if is_training:  # Training

                with tf.GradientTape() as tape:

                    fes = np.zeros(nem.shape)
                    out = model(np.concatenate([np.expand_dims(nem[:, :, :], -1), np.expand_dims(fes[:, :, :], -1)], axis=-1))
                    out = tf.squeeze(out)

                    loss = self.criterion(out, y)

                grads = tape.gradient(loss, model.trainable_weights)

                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

            else:  # Validation

                fes = np.zeros(nem.shape)
                out = model(np.concatenate([np.expand_dims(nem[:, :, :], -1), np.expand_dims(fes[:, :, :], -1)], axis=-1))
                out = tf.squeeze(out)

                loss = self.criterion(out, y)

            # Get metric
            out_acc = np.zeros(y.shape)
            out_acc[out.numpy() > .5] = 1
            metric = sum(y-out_acc == 0) / y.shape

            # Track train loss by multiplying average loss by number of examples in batch
            loss_over_all += loss.numpy()
            metric_over_all += metric[0]

            # Track training progress
            progress_bar(ii + 1, len(generator), loss.numpy(), metric[0], timer() - self.start_time_epoch)

        # Calculate average loss and metrics
        loss_over_all = loss_over_all / len(generator)
        metric_over_all = metric_over_all / len(generator)

        return metric_over_all, loss_over_all


def progress_bar(batch, total_batches, metric, loss, eta, metric_val='', loss_val=''):
    n = 40
    batches_per_step = max(total_batches // n, 1)
    eta = str(datetime.timedelta(seconds=eta))[2:7]

    bar = 'Batch ' + str(batch) + '/' + str(total_batches) + ' -- ['

    for i in range((batch // batches_per_step) + 1):
        bar = bar + '='
    bar = bar + '>'
    for ii in range(n - (batch // batches_per_step + 1)):
        bar = bar + '.'

    if metric_val == '':
        bar = bar + '] -- metric: ' + str(round(metric, 4)) + ' -- loss: ' + str(round(loss, 4)) + ' -- ETA: ' + eta
        print(bar, end='\r')
    else:
        bar = bar + '] -- metric: ' + str(round(metric, 4)) + ' -- loss: ' + str(
            round(loss, 4)) + ' -- val_metric: ' + str(round(metric_val, 4)) + ' -- val_loss: ' + str(
            round(loss_val, 4)) + ' -- ETA: ' + eta
        print(bar)


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=100, lr0=0.001):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch < lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr0 * np.exp(-0.25 * (epoch - lr_decay_epoch))

    return optimizer

############################################
# DIGITAL SIGNAL PROCESSING
############################################


def writewav(f, sr, x):

    xdesn = 2**15*x
    xint16 = xdesn.astype('int16')
    wavfile.write(f, sr, xint16)


def readwav(f):
    fs1, x1 = wavfile.read(f)
    xx1 = x1.astype('float64')
    xx1 *= 2 ** (-15)
    return fs1, xx1


def norm_spectrogram(stft, w_size, fs, mel=False):
    MD = 50
    ep = 10**(-1*MD/20)

    stft_norm = 20*np.log10(np.abs(stft)/(w_size/2)+ep)
    stft_norm = (stft_norm+MD)/MD

    if mel:
        stft_norm = stft_to_mel(stft_norm, fs)

    return stft_norm


def denorm_spectrogram(stft_norm, w_size, fs, mel=False):
    MD = 50
    ep = 10**(-1*MD/20)

    if mel:
        stft_norm = mel_to_stft(stft_norm, fs)

    stft = (stft_norm*MD)-MD
    stft = (10**(stft/20)-ep)*(w_size/2)

    return stft


def spectrogram(s, fs, mel=False):
    s = s.astype('float16')
    w_size_ms = 20
    w_size_n = int(w_size_ms * (1e-3) * fs)
    olap = 0.75

    stft = librosa.core.stft(s, n_fft=w_size_n, hop_length=int(w_size_n-w_size_n*(olap)), win_length=w_size_n, window='hann',
                             center=True)
    stft = stft[:-1, :]

    stft_m = norm_spectrogram(stft, w_size_n, mel=mel)

    stft_p = np.angle(stft)

    return stft_m, stft_p


def recover_from_spectrogram(stft_m, stft_p, fs, mel=False):
    w_size_ms = 20
    w_size_n = int(w_size_ms * (1e-3) * fs)
    olap = 0.75

    # Inverse operations to denormalize stft
    stft_m = denorm_spectrogram(stft_m, w_size_n, fs, mel=mel)

    # Combining phase and signal
    stft = stft_m.astype(np.complex) * np.exp(1j * stft_p)

    # Freqs padding
    stft = np.concatenate([stft, np.ones((1, stft.shape[1]))*complex(0, 0)], axis=0)

    # Time padding

    # IFFT
    s = librosa.istft(stft, hop_length=int(w_size_n-w_size_n*olap), win_length=w_size_n)

    s = s.astype('float32')

    return s


def mel_to_stft(stft, fs):

    (NFFT, Nbin_t) = stft.shape

    # Eje de frecuencias discretas
    fd = np.arange(0, NFFT+1)/NFFT
    # eje de frecuencias analógicas
    fa_p = (fs/2)*fd
    # Número de frecuencias a considerar
    Npuntos = fa_p.shape[0]

    # Mallado en el eje de frecias Mel
    fmelgrid = np.arange(0, Npuntos)/(Npuntos-1) * 2595*np.log10(1+fa_p[-1]/700)
    # Frecuencias lineales correspondientes
    fgrida=700*(10**(fmelgrid/2595)-1)

    # Mallado para generar los anchos freceunciales en frecuencias MEL que corresponden a cada bin frecuencial lineal
    flinealgrid = np.arange(0, 2*Npuntos-1)/(2*Npuntos-1-1) * 700*(10**(fmelgrid[-1]/2595)-1)
    flinealgrid = np.concatenate([np.expand_dims(flinealgrid[0], axis=0), flinealgrid[1:-1][::2], np.expand_dims(flinealgrid[-1], axis=0)])

    X = np.zeros((NFFT+1, Nbin_t))
    for k in np.arange(1, flinealgrid.shape[0]-2):
        indices = np.argwhere([list(fgrida > flinealgrid[k - 1])[j] and list(fgrida < flinealgrid[k])[j] for j in range(fgrida.__len__())])
        if len(indices) == 0:
            X[k,:] = X[k-1, :]
        else:
            if len(indices) > 1:
                X[k,:]=np.mean(np.squeeze(stft[indices, :]),axis=0)
            else:
                X[k, :] = np.squeeze(stft[indices, :])
    X = X[1:, :]
    return X


def stft_to_mel(stft, fs):

    (NFFT, Nbin_t) = stft.shape

    # Eje de frecuencias discretas
    fd = np.arange(0, NFFT)/NFFT
    # eje de frecuencias analógicas
    fa_p = (fs/2)*fd
    # Número de frecuencias a considerar
    Npuntos = fa_p.shape[0]

    # Mallado en el eje de frecias Mel
    fmelgrid = np.arange(0, Npuntos)/Npuntos * 2595*np.log10(1+fa_p[-1]/700)
    # Frecuencias lineales correspondientes
    fgrida=700*(10**(fmelgrid/2595)-1)

    # Indices redondeados (donde buscar en la fft a esas frecuencias)
    indices = np.round(fgrida*NFFT/(fs/2))

    Xmel = np.concatenate([np.expand_dims(stft[int(i), :], 0) for i in indices])

    return Xmel


def double_talk_detector(stft_ne, stft_fe, NFFT, fs, threshold=20):

    # Determinar bandas frecuenciales de interés
    ejef = np.arange(0, NFFT) / NFFT * (fs)
    idx0 = np.argmin(np.abs(ejef - 300))
    idxf = np.argmin(np.abs(ejef - 5000))

    # Obtener energías en esas bandas para ambos nodos
    E_nes_i = 10 * np.log10(np.maximum(np.sum(np.power(stft_ne[idx0:idxf], 2)), 10e-11))
    E_fes_i = 10 * np.log10(np.maximum(np.sum(np.power(stft_fe[idx0:idxf], 2)), 10e-11))

    # Umbralizar
    if E_nes_i > threshold and E_fes_i > threshold:
        is_dt = True
    else:
        is_dt = False

    return is_dt, E_nes_i, E_fes_i