from PIL import Image
import pandas as pd
# from sklearn.utils import shuffle
from time import time
import numpy as np
from sklearn.metrics import fbeta_score
import sys

wd = 'E:/new_data/kaggle/planet/'

def myinit(wd):
    train_set = pd.read_csv(wd + 'train_v2.csv')
    train_set['tags'] = train_set['tags'].apply(lambda x: x.split(' '))
    test_set = pd.read_csv(wd+'sample_submission_v2.csv')
    train_tags = ['clear', 'partly_cloudy', 'haze', 'cloudy', 'primary', 'agriculture', 
        'road', 'water', 'cultivation', 'habitation', 'bare_ground', 'selective_logging', 
        'artisinal_mine', 'blooming', 'slash_burn', 'conventional_mine', 'blow_down']
    label_map = {l: i for i, l in enumerate(train_tags)}
    inv_label_map = {i: l for l, i in label_map.items()}
    file_all = train_set['image_name'].values
    y_tr = get_y(train_set['tags'].values, label_map)
    test_file_all = test_set['image_name'].values
    return wd, train_set, test_set, label_map, inv_label_map, file_all, y_tr, test_file_all

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

def f2_opti_score_new(y_true, y_pred, thresholds = np.arange(0, 1, 0.05), num_classes=17):
    x = np.zeros(num_classes)
    for i in range(num_classes):
        best_j, best_score = 0, 0
        for j, t in enumerate(thresholds):
            pred = y_pred[:, i] > t
            score = fbeta_score(y_true[:, i], pred, beta = 2)
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

if sys.version[:3] == '3.5':
    from scipy.stats import kurtosis
    from scipy.stats import skew

    def extract_features(dir_path):
        img_paths = os.listdir(dir_path)
        N = len(img_paths)
        features = np.array([]).reshape((-1, 25))
        for i, f in enumerate(tqdm(img_paths, mininterval=1, ncols=80)): 
            im = np.array(Image.open(dir_path + f).convert('RGB'))
            r = im[:,:,0].ravel()
            g = im[:,:,1].ravel()
            b = im[:,:,2].ravel()
            fetr = np.array([r.mean(), g.mean(), b.mean(), r.std(), g.std(), b.std(), 
                    r.max(), g.max(), b.max(), r.min(), g.min(), b.min(),
                    r.max() - r.min(), g.max() - g.min(), b.max() - b.min(), 
                    kurtosis(r), kurtosis(g), kurtosis(b), skew(r), skew(g), skew(b),
                    im.mean(), im.std(), im.max(), im.min()
                    ]).reshape((-1, 25))
            features =  np.concatenate((features, fetr))
        return features

    def get_x(_dir):
        if 'train' in _dir:
            tfidf_file = './despts/tr_tfidf_256.npy'
            data_set = train_set
        else:
            tfidf_file = './despts/ts_tfidf_256.npy'
            data_set = test_set
        features = extract_features(_dir)
        tfidf = np.load(tfidf_file)
        ftr = np.c_[features, tfidf]
        df = pd.DataFrame(ftr)
        df['image_name'] = list(map(lambda f: f[:-4], os.listdir(_dir)))
        X = data_set.merge(df, on='image_name').iloc[:, 2:].values
        return X

    def  get_weight(y):
        w0 = y.mean() # 0类的权重用1类的占比
        w1 = 1 - w0
        w = np.zeros_like(y, dtype='float')
        w[y==0.] = w0
        w[y==1.] = w1
        return w

if sys.version[:3] == '3.6':
    from torch.utils.data.sampler import WeightedRandomSampler
    from torch.utils.data import DataLoader
    import torch
    from mydataset import PlanetDataset

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