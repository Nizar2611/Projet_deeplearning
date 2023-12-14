# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 20:39:51 2022

@author: User
"""
## Biblio:
    
# Pour les multiprocesseurs
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import joblib

# Pour la partie entrainement
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Initializer, LRScheduler, TensorBoard
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# Pour la partie pre-entrainement et traitement de signal
from pathlib import Path
import scipy.signal as sg
import wfdb
import pywt
import cv2 
import numpy as np 




PATH = Path("dataset")
sampling_rate = 360

# les labels à suprimer 
invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']




def preprocess(record):
    # Lecture des signaux 
    signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[0]).p_signal[:, 0]
    annotation = wfdb.rdann((PATH / record).as_posix(), extension="atr")
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)

    # Filtrage de la baseline
    baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    filtered_signal = signal - baseline

    # nettoyage de non-beat label
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
    r_peaks, labels = r_peaks[indices], labels[indices]

    # normalization de signal filtré
    normalized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])

    # AAMI categories
    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB
        "V": 2, "E": 2,  # VEB
        "F": 3,  # F
        "/": 4, "f": 4, "Q": 4  # Q
    }
    categories = [AAMI[label] for label in labels]

    return {
        "record": record,
        "signal": normalized_signal, "r_peaks": r_peaks, "categories": categories
    }



# Préparation des datasets
if __name__ == "__main__":
    # Pour multiprocesseur
    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1

    train_records = [
        '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
        '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
        '223', '230'
    ]
    print("train processing...")
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        train_data = [result for result in executor.map(preprocess, train_records)]

    test_records = [
        '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
        '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
        '233', '234'
    ]
    print("test processing...")
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        test_data = [result for result in executor.map(preprocess, test_records)]

    print("ok!")




def worker(data, wavelet, scales, sampling_period):
    # convertion en scalogramme
    avant, apres = 90, 110
    coeffs, frequencies = pywt.cwt(data["signal"], scales, wavelet, sampling_period)
    r_peaks, categories = data["r_peaks"], data["categories"]

    
    x, y, groups = [], [], []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue

        if categories[i] == 4:  # on suprime la classe Q 
            continue

        # redimensionement (100*100) 
        x.append(cv2.resize(coeffs[:, r_peaks[i] - avant: r_peaks[i] + apres], (100, 100)))
        
        y.append(categories[i])
        groups.append(data["record"])

    return x, y, groups


# Preparation de la data pour l'entree du CNN
def load_data(wavelet, scales, sampling_rate):

    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1 

    # Entrainement
    x_train, y_train, groups_train = [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x, y, groups in executor.map(
                partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), train_data):
            x_train.append(x)
            
            y_train.append(y)
            groups_train.append(groups)
            
            
    x_train = np.expand_dims(np.concatenate(x_train, axis=0), axis=1).astype(np.float32)
    
    y_train = np.concatenate(y_train, axis=0).astype(np.int64)
    groups_train = np.concatenate(groups_train, axis=0)

    # test
    x_test, y_test, groups_test = [], [], []
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        for x, y, groups in executor.map(
                partial(worker, wavelet=wavelet, scales=scales, sampling_period=1. / sampling_rate), test_data):
            x_test.append(x)
            y_test.append(y)
            groups_test.append(groups)

    x_test = np.expand_dims(np.concatenate(x_test, axis=0), axis=1).astype(np.float32)
    y_test = np.concatenate(y_test, axis=0).astype(np.int64)
    groups_test = np.concatenate(groups_test, axis=0)

    
 
    return (x_train, y_train, groups_train), (x_test, y_test, groups_test)

#definition de NN
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 7)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pooling1 = nn.MaxPool2d(5)
        self.pooling2 = nn.MaxPool2d(3)
        self.pooling3 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(64, 32) 
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (16 x 94 x 94)
        x = self.pooling1(x)  # (16 x 18 x 18)
        x = F.relu(self.bn2(self.conv2(x)))  # (32 x 16 x 16)
        x = self.pooling2(x)  # (32 x 5 x 5)
        x = F.relu(self.bn3(self.conv3(x)))  # (64 x 3 x 3)
        x = self.pooling3(x)  # (64 x 1 x 1)
        x = x.view((-1, 64))  # (64,)
        x = F.relu(self.fc1(x))  # (32,)
        x = self.fc2(x)  # (4,)
        return x


def main():
    sampling_rate = 360

    wavelet = "mexh"  # mexh (morl ?)
    scales = pywt.central_frequency(wavelet) * sampling_rate / np.arange(1, 101, 1)

    (x_train, y_train, groups_train), (x_test, y_test, groups_test) = load_data(
        wavelet=wavelet, scales=scales, sampling_rate=sampling_rate)
    print("Data loaded successfully!")

    log_dir = "./logs/log"
    shutil.rmtree(log_dir, ignore_errors=True)

    callbacks = [
        Initializer("[conv|fc]*.weight", fn=torch.nn.init.kaiming_normal_),
        Initializer("[conv|fc]*.bias", fn=partial(torch.nn.init.constant_, val=0.0)),
        LRScheduler(policy=StepLR, step_size=5, gamma=0.1),
        EpochScoring(scoring=make_scorer(f1_score, average="macro"), lower_is_better=False, name="valid_f1"),
        TensorBoard(SummaryWriter(log_dir))
    ]
    net = NeuralNetClassifier(  
        MyModule,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        max_epochs=30,
        batch_size=1024,
        train_split=predefined_split(Dataset({"x": x_test}, y_test)),
        verbose=1,
        device="cpu",
        callbacks=callbacks,
        iterator_train__shuffle=True,
        optimizer__weight_decay=0,
    )
    net.fit({"x": x_train}, y_train)
    y_true, y_pred = y_test, net.predict({"x": x_test})
    

    print(net)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
    



if __name__ == "__main__":
    main()