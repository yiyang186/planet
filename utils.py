from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from time import time
import numpy as np
import torch
from mydataset import PlanetDataset
from sklearn.metrics import fbeta_score

wd = 'E:/new_data/kaggle/planet/'

def weighted_from_probs(probs, y_all):
    tags_sum = y_all.sum(axis=0)
    probs = zip(range(len(probs)), probs, tags_sum)
    probs = sorted(probs, key=lambda item: item[2], reverse=True)
    weights = np.zeros(y_all.shape[0])
    for i, w, _ in probs:
        weights[y_all[:, i] > 0] = w
    return weights

def weighted_train_loader(x, y, probs, transf, batch=256, pic_size=(128, 128), factor=None):
    weights = weighted_from_probs(probs, y)
    if not factor is None:
        weights = weights * factor
    sampler = WeightedRandomSampler(weights, y.shape[0])

    train_dataset = PlanetDataset(files=x, labels=y, transform=transf, pic_size=pic_size)
    trainloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, sampler=sampler)
    return trainloader

def valid_loader(x, y, transf, batch=256, pic_size=(128, 128)):
    valie_dataset = PlanetDataset(files=x, labels=y, transform=transf, pic_size=pic_size)
    valieloader = DataLoader(valie_dataset, batch_size=batch, shuffle=False, sampler=None)
    return valieloader

def test_loader(x, transf, batch=256, pic_size=(128, 128)):
    test_dataset = PlanetDataset(files=x, test=True, transform=transf, pic_size=pic_size)
    testloader = DataLoader(test_dataset, batch_size=batch, shuffle=False, sampler=None)
    return testloader
        
def get_y(tags, label_map):
    y = np.zeros((tags.shape[0], 17), dtype=np.uint8)
    for i, tag in enumerate(tags):
        for t in tag:
            y[i, label_map[t]] = 1
    return y

def f2_score(y_true, y_prob, thresholds = None, num_classes=17):
    if thresholds is None:
        score = fbeta_score(y_true, y_prob, beta = 2, average = 'samples')
    elif len(thresholds) == num_classes:
        y_pred = np.zeros_like(y_prob, dtype = np.int8)
        for i in range(num_classes):
            y_pred[:, i] = y_prob[:, i] > thresholds[i]
        score = fbeta_score(y_true, y_pred, beta = 2, average='samples')
    return score

def f2_opti_score(y_true, y_pred, thresholds = np.arange(0, 1, 0.05), num_classes=17):
    x = np.zeros(num_classes)
    for i in range(num_classes):
        best_j, best_score = 0, 0
        for j, t in enumerate(thresholds):
            x[i] = t
            score = f2_score(y_true, y_pred, x, num_classes=num_classes)
            if score > best_score:
                best_j = j
                best_score = score
        x[i] = thresholds[best_j]
    return x

def to_submit(y_prob, th, test_df, inv_label_map):
    y_pred = np.zeros_like(y_prob, dtype=np.int8)
    for i in range(17):
        y_pred[:, i] = y_prob[:, i] > th[i]
    for i in range(test_df.shape[0]):
        test_df.iloc[i, 1] = ' '.join([inv_label_map[x] for x in np.where(y_pred[i] == 1)[0]])
    return test_df

def to_01(y_prob, th):
    y_pred = np.zeros_like(y_prob, dtype=np.int8)
    for i in range(17):
        y_pred[:, i] = y_prob[:, i] > th[i]
    return y_pred


