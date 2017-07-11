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

def weighted_train_loader(x, y, probs, transf, batch=256, pic_size=(128, 128)):
    weights = weighted_from_probs(probs, y)
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

def generate_train_set(x, y, transf, batch=256, pic_size=(128, 128)):
    gy = torch.zeros(batch, y.shape[0])
    gx = torch.zeros(batch, 3, pic_size[0], pic_size[1])
    st = 0
    while st + batch < x.shape[0]:
        for gi, _x in enumerate(x[st: st + batch]):
            img = Image.open(wd+'train-jpg/{}.jpg'.format(_x)).convert('RGB')
            gx[gi] = transf(img, pic_size)
        gy = torch.from_numpy(y[st: st + batch])
        st += batch
        yield (gx, gy)
    for gi, _x in enumerate(x[st:]):
        img = Image.open(wd+'train-jpg/{}.jpg'.format(_x)).convert('RGB')
        gx[gi] = transf(img, pic_size)
    gx = gx[: gi+1]
    gy = torch.from_numpy(y[st:])
    return (gx, gy)
        
def generate_test_set(x, transf, batch=256, pic_size=(128, 128)):
    gx = torch.zeros(batch, 3, pic_size[0], pic_size[1])
    st = 0
    while st + batch < x.shape[0]:
        for gi, _x in enumerate(x[st: st + batch]):
            img = Image.open(wd+'test-jpg/{}.jpg'.format(_x)).convert('RGB')
            gx[gi] = transf(img.resize(pic_size))
        st += batch
        yield gx
    for gi, _x in enumerate(x[st:]):
        img = Image.open(wd+'test-jpg/{}.jpg'.format(_x)).convert('RGB')
        gx[gi] = transf(img.resize(pic_size))
    gx = gx[: gi+1]
    return gx
        
def get_y(tags, label_map):
    y = np.zeros((tags.shape[0], 17), dtype=np.uint8)
    for i, tag in enumerate(tags):
        for t in tag:
            y[i, label_map[t]] = 1
    return y

def f2_score(label, pred, thresholds = None):
    if thresholds is None:
        score = fbeta_score(label, pred, beta = 2, average = 'samples')
    elif len(thresholds) == 17:
        p = np.zeros_like(pred, dtype = np.int8)
        for j in range(17):
            p[:, j] = pred[:, j] > thresholds[j]
        score = fbeta_score(label, p, beta = 2, average='samples')
    return score

# 找到最优th
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

def to_submit_new(y_prob, th, test_df):
    y_pred = np.zeros_like(y_prob, dtype=np.int8)
    f4 = y_prob[:, :4].argmax(axis=1)
    for i in range(4, 17):
        y_pred[:, i] = y_prob[:, i] > th[i]
    for i in range(test_df.shape[0]):
        test_df.iloc[i, 1] = ' '.join([inv_label_map[f4[i]]] + [inv_label_map[x] for x in np.where(y_pred[i] == 1)[0]])
    return test_df

def to_submit(y_prob, th, test_df, inv_label_map):
    y_pred = np.zeros_like(y_prob, dtype=np.int8)
    for i in range(17):
        y_pred[:, i] = y_prob[:, i] > th[i]
    for i in range(test_df.shape[0]):
        test_df.iloc[i, 1] = ' '.join([inv_label_map[x] for x in np.where(y_pred[i] == 1)[0]])
    return test_df

def to_01(y_prob, th, test_df):
    y_pred = np.zeros_like(y_prob, dtype=np.int8)
    for i in range(17):
        y_pred[:, i] = y_prob[:, i] > th[i]
    return y_pred 

def f2_score(y_true, y_prob, thresholds = None, num_classes=17):
    if thresholds is None:
        score = fbeta_score(y_true, y_prob, beta = 2, average = 'samples')
    elif len(thresholds) == num_classes:
        y_pred = np.zeros_like(y_prob, dtype = np.int8)
        for i in range(num_classes):
            y_pred[:, i] = y_prob[:, i] > thresholds[i]
        score = fbeta_score(y_true, y_pred, beta = 2, average='samples')
    return score

# 根据模型训练历史绘制学习曲线
def plot_learning_curve(history, param='loss'):
    plt.figure(figsize=(10, 8))
    plt.plot(history.history[param])
    plt.plot(history.history['val_'+param])
    plt.title('model '+param)
    plt.ylabel(param)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
