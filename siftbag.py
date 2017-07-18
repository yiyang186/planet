import cv2
import imp
import os
import gc
from time import time
import numpy as np
from tqdm import tqdm

tr_dir = 'E:/new_data/kaggle/planet/train-jpg/'
ts_dir = 'E:/new_data/kaggle/planet/test-jpg/'
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# 抽取训练集的描述子
tr_files = os.listdir(tr_dir)
tr_despts = np.array([]).reshape((-1, 128))

for i, f in enumerate(tqdm(tr_files, mininterval=1, ncols=80)):
    if i > 0 and i % 1000 == 0:
        np.save('./despts/tr_despts_{}.npy'.format(i), tr_despts)
        tr_despts = np.array([]).reshape((-1, 128))
        gc.collect()
              
    img_path = tr_dir + f
    img = cv2.imread(img_path)
    kps = fea_det.detect(img)
    kps, despt = des_ext.compute(img, kps)
    if not despt is None:
        tr_despts = np.concatenate((tr_despts, despt), axis=0)
np.save('./despts/tr_despts_{}.npy'.format(i), tr_despts)

# 抽取测试集的描述子
ts_files = os.listdir(ts_dir)
ts_despts = np.array([]).reshape((-1, 128))

for i, f in enumerate(tqdm(ts_files, mininterval=1, ncols=80)):
    if i > 0 and i % 1000 == 0:
        np.save('./despts/ts_despts_{}.npy'.format(i), ts_despts)
        ts_despts = np.array([]).reshape((-1, 128))
        gc.collect()
              
    img_path = ts_dir + f
    img = cv2.imread(img_path)
    kps = fea_det.detect(img)
    kps, despt = des_ext.compute(img, kps)
    if not despt is None:
        ts_despts = np.concatenate((ts_despts, despt), axis=0)
np.save('./despts/ts_despts_{}.npy'.format(i), ts_despts)

# 导入上面写入磁盘中的所有描述子
despts = np.array([]).reshape((-1, 128))
for i, f in enumerate(tqdm(os.listdir('./despts/'), mininterval=1, ncols=80)):
    dpt = np.load('./despts/' + f)
    despts = np.concatenate((despts, dpt), axis=0)

# 对所有描述子做kmeans聚类，聚成256类
from scipy.cluster.vq import kmeans
st = time()
voc, variance = kmeans(despts, 256)
print(time()-st) # 用时14个小时
np.save('./despts/kmeans256_centers_var{}.npy'.format(int(variance)), voc)

# 提取训练集的描述子转化成tf-idf向量
from scipy.cluster.vq import vq

tr_files = os.listdir(tr_dir)
tr_despts = np.array([]).reshape((-1, 128))

tf = np.zeros((len(tr_files), 256), dtype='float64')
for i, f in enumerate(tqdm(tr_files, mininterval=1, ncols=80)):         
    img_path = tr_dir + f
    img = cv2.imread(img_path)
    kps = fea_det.detect(img)
    kps, despt = des_ext.compute(img, kps)
    if not despt is None:
        words, distance = vq(despt, voc)
        for w in words:
            tf[i][w] += 1.0
occurences = (tf > 0).sum(axis = 0)
idf = np.log(float(len(tr_files)) / occurences.astype('float64'))
tfidf = tf * idf
np.save('./despts/tr_tfidf_256.npy', tfidf)

# 提取测试集的描述子转化成tf-idf向量
from scipy.cluster.vq import vq

ts_files = os.listdir(ts_dir)
ts_despts = np.array([]).reshape((-1, 128))

ts_tf = np.zeros((len(ts_files), 256), dtype='float64')
for i, f in enumerate(tqdm(ts_files, mininterval=1, ncols=80)):
    img_path = ts_dir + f
    img = cv2.imread(img_path)
    kps = fea_det.detect(img)
    kps, despt = des_ext.compute(img, kps)
    if not despt is None:
        words, distance = vq(despt, voc)
        for w in words:
            ts_tf[i][w] += 1.0
np.save('./despts/ts_tf_256.npy', ts_tf)
ts_occurences = (ts_tf > 0).sum(axis = 0)
ts_idf = np.log(float(len(ts_files)) / ts_occurences.astype('float64'))
ts_tfidf = ts_tf * ts_idf
np.save('./despts/ts_tfidf_256.npy', ts_tfidf)