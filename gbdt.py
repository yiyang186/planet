import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import utils
import imp
imp.reload(utils)

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
y_train = utils.get_y(train_set['tags'].values, label_map)
test_file_all = test_set['image_name'].values

tr_dir = 'E:/new_data/kaggle/planet/train-jpg/'
ts_dir = 'E:/new_data/kaggle/planet/test-jpg/'

# 获取带有sift描述子的数据集
X_train= utils.get_x(ts_dir)
X_test = utils.get_x(ts_dir)

# 训练
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold # 分层交叉验证
from sklearn.metrics import fbeta_score

p_tr = np.zeros((X_train.shape[0], 17))
y_ts = np.zeros((X_test.shape[0], 17))
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'device': 'gpu',
    'verbosity': -1
}

num_classes = 17
n_splits = 5
for i_c in range(num_classes):
    skf = StratifiedKFold(n_splits=n_splits)
    w = utils.get_weight(y_train[:, i_c])
    k_now = 0
    for i_tr, i_vl in skf.split(X_train, y_train[:, i_c]):
        print(i_c, k_now)
        lgb_train = lgb.Dataset(X_train[i_tr], y_train[i_tr, i_c], weight=w[i_tr], free_raw_data=False)
        lgb_eval = lgb.Dataset(X_train[i_vl], y_train[i_vl, i_c], reference=lgb_train, weight=w[i_vl], free_raw_data=False)
        bst = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=lgb_eval, early_stopping_rounds=30)
        p_tr[i_vl, i_c]= bst.predict(X_train[i_vl], num_iteration=bst.best_iteration)
        y_ts[:, i_c] = bst.predict(X_test, num_iteration=bst.best_iteration) / float(n_splits)
        k_now += 1


th1 = utils.f2_opti_score(y_train, p_tr, thresholds = np.arange(0, 1, 0.01))
th2 = utils.f2_opti_score(y_train, p_tr, thresholds = np.arange(1, 0, -0.01))
th = (th1 + th2) / 2.0
print(utils.f2_score(y_train, p_tr, th, num_classes=17))

# 保存
np.save('./pred/modelsiftgdbt_date719_pred_train.npy', p_tr)
np.save('./pred/modelsiftgdbt_date719_pred_test.npy', y_ts)