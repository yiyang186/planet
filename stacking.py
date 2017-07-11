import torch
import numpy as np
import pandas as pd
import skimage
import matplotlib.pyplot as plt
import imp
import core
import utils
%matplotlib inline

# 初始化
wd = 'E:/new_data/kaggle/planet/'
train_set = pd.read_csv(wd + 'train_v2.csv')
train_set['tags'] = train_set['tags'].apply(lambda x: x.split(' '))
test_set = pd.read_csv(wd+'sample_submission_v2.csv')
train_tags = ['clear', 'partly_cloudy', 'haze', 'cloudy', 'primary', 'agriculture', 'road', 'water',
             'cultivation', 'habitation', 'bare_ground', 'selective_logging', 'artisinal_mine', 
              'blooming', 'slash_burn', 'conventional_mine', 'blow_down']
label_map = {l: i for i, l in enumerate(train_tags)}
inv_label_map = {i: l for l, i in label_map.items()}
file_all = train_set['image_name'].values
y_tr = utils.get_y(train_set['tags'].values, label_map)
test_file_all = test_set['image_name'].values

# 加载验证和预测结果
nnts = np.load('../preds/170620_01_pred_ts.npy')
nntr = np.load('../preds/170620_01_pred_tr.npy')

gbts = np.load('../preds/gbm_all_pred_ts_170624_01.npy')
gbtr = np.load('../preds/gbm_all_pred_tr_170624_01.npy')

wnnts = np.load('./pred/modelweightedNN1_date710_pred_test.npy')
wnntr = np.load('./pred/modelweightedNN1_date710_pred_train.npy')

# stacking
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold

y_ts = np.zeros((gbts.shape[0], 17))
p_tr = np.zeros((gbtr.shape[0], 17))

for i_c in range(17):
    skf = StratifiedKFold(n_splits=5)
    x_tr = np.c_[nntr[:, i_c], gbtr[:, i_c], wnntr[:, i_c]]
    x_ts = np.c_[nnts[:, i_c], gbts[:, i_c], wnnts[:, i_c]]
    k_now = 0
    for i_tr, i_vl in skf.split(x_tr, y_tr[:, i_c]):
        lr = LogisticRegression(class_weight='balanced')
        lr.fit(x_tr[i_tr, :], y_tr[i_tr, i_c])
        p_tr[i_vl, i_c] = lr.predict_proba(x_tr[i_vl, :])[:,1]

    lr = LogisticRegression(class_weight='balanced')
    lr.fit(x_tr, y_tr[:, i_c])
    y_ts[:, i_c] = lr.predict_proba(x_ts)[:, 1]

# 求最佳阈值
th1 = utils.f2_opti_score(y_tr, p_tr, thresholds = np.arange(0, 1, 0.01))
th2 = utils.f2_opti_score(y_tr, p_tr, thresholds = np.arange(1, 0, -0.01))
th = (th1 + th2) / 2.0
utils.f2_score(y_tr, p_tr, th, num_classes=17)

# 输出结果
submit_df = utils.to_submit(y_ts, th, test_set, inv_label_map)
submit_df.to_csv('./submit/3model_stacking_date711_no2.csv', index=False)

submit_df1 = utils.to_submit_new(y_ts, th, test_set, inv_label_map)
submit_df1.to_csv('./submit/3model_stacking_date711_no2.csv', index=False)