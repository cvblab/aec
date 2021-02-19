########################################################
# Imports
########################################################

import random
import numpy as np
import pandas as pd
from utils import *

# GPU selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(42)
np.random.seed(0)

########################################################
# Hyperparams - AEC MODEL TRAINING
########################################################


dir_dataset = '../datasets/synthetic/'
dir_results = '../results/mel_perceptual/'
architecture = 'unet'
method = 'aec_perceptual'
n_epochs = 30
lr = 5*1e-4
batch_size = 128
opt = 'nadam'

debugging = True
train_on_gpu = True
device = []

########################################################
# Main training
########################################################

# Prepare data generators
train_dataset = Dataset(dir_dataset, partition='train', debugging=debugging)
train_loader = DataGenerator(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = Dataset(dir_dataset, partition='test', debugging=debugging)
test_loader = DataGenerator(test_dataset, batch_size=batch_size, shuffle=True)

# Get cnnMOS model
cnnMOS, _, _ = CNN_MOS(input_size=(160, 32), optimizer=opt, learning_rate=lr, mode=2,
                       number_filters_0=64, resize_factor_0=[1, 1])
# Load weights
cnnMOS.load_weights('qualityMOS/model_best')
# Extract feature space
cnnMOS = tf.keras.Model(inputs=cnnMOS.layers[0].input, outputs=cnnMOS.layers[-3].output)

# Get aec model
model, criterion, optimizer = u_net_2d(input_size=(160, 32), optimizer=opt, learning_rate=lr, mode=2,
                                       number_filters_0=16, resize_factor_0=[1, 1])

# Initialize trainer object
trainer = CNNTrainer(n_epochs, criterion, optimizer, train_on_gpu, device, dir_results,
                     save_best_only=True, normalization=None, other_model=cnnMOS, method=method)

model, history = trainer.train(model, train_generator=train_loader, val_generator=test_loader)











