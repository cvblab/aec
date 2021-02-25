

########################################################
# Imports
########################################################
from __future__ import print_function

import random
import numpy as np
import pandas as pd
from utils import *
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from hyperopt import Trials, STATUS_OK, tpe
import keras, sys, h5py
import json

from hyperas import optim
from hyperas.distributions import choice, uniform


# GPU selection
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

random.seed(42)
np.random.seed(0)

########################################################
# Hyperparams - AEC MODEL TRAINING
########################################################






# Prepare data generators
def data():
    dir_dataset = '../datasets/synthetic/'
    #dir_dataset = '/scratch/eco/synthetic/'
    batch_size = 128
    debugging = True

    train_dataset = Dataset(dir_dataset, partition='train', debugging=debugging)
    train_loader = DataGenerator(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Dataset(dir_dataset, partition='test', debugging=debugging)
    test_loader = DataGenerator(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def model(train_loader, test_loader):
    train_on_gpu = True
    dir_results = '../results/mel/'
    architecture = 'transformer'
    method = 'aec'
    n_epochs = 2
    device = []

    # Get model
    choice_lr = {{choice([1e-3, 1e-4])}}
    choice_opt = {{choice(['nadam', 'sgd', 'adam'])}}
    choice_number_layers = {{choice([8, 12, 16])}}
    choice_number_heads = {{choice([6, 8, 10])}}
    choice_dropout = {{choice([0.1, 0.2, 0.5])}}

    model, criterion, optimizer = speechenhancement_transformer_2d(input_size=(160, 32), optimizer=choice_opt,
                                                                   learning_rate=choice_lr, mode=2, number_layers=choice_number_layers, 
                                                                   number_heads=choice_number_heads, dropout=choice_dropout)



    # Initialize trainer object
    trainer = CNNTrainer(n_epochs, criterion, optimizer, train_on_gpu, device, dir_results,
                        save_best_only=True, normalization=None, method=method)

    model, history = trainer.train(model, train_generator=train_loader, val_generator=test_loader)

    return {'loss': history['loss_val'], 'status': STATUS_OK}


trials = Trials()

best_run, best_model = optim.minimize(model=model,
                                      data=data,
                                      algo=tpe.suggest,
                                      max_evals=100,
                                      trials=trials)

f=open("best_model.json","w")
f.write(best_model.to_json())
f.close()
print("Best performing model chosen hyper-parameters:")
print(best_run)
f=open("best_run.json","w")
json.dump(best_run, f)
f.close()

