import torch
import numpy as np
import pandas as pd
import skimage
import imp
import core
import utils

wd = 'E:/new_data/kaggle/planet/'
train_set = pd.read_csv(wd + 'train_v2.csv')
train_set['tags'] = train_set['tags'].apply(lambda x: x.split(' '))
test_set = pd.read_csv(wd+'sample_submission_v2.csv')
train_tags = ['clear', 'partly_cloudy', 'haze', 'cloudy', 'primary', 'agriculture', 'road', 'water',
             'cultivation', 'habitation', 'bare_ground', 'selective_logging', 'artisinal_mine', 
              'blooming', 'slash_burn', 'conventional_mine', 'blow_down']
label_map = {l: i for i, l in enumerate(train_tags)}
inv_label_map = {i: l for l, i in label_map.items()}

# 超参数
batch_size = 256
pic_size = (64, 64)
learning_rate = 1e-3
num_epoches = 1000
tolerance = 15
lr_tolerance = 7
transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
])
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def transform_tr(img, pic_size):
    img = img.resize(pic_size)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)  if random.randint(0, 1) > .5 else img
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  if random.randint(0, 1) > .5 else img
    img.rotate(np.random.random() * 45)
    img_tensor = transform2(img)
    return img_tensor

def transform_vl(img, pic_size):
    img = img.resize(pic_size)
    img_tensor = transform2(img)
    return img_tensor

n_splits = 5
model_name = 'weightedNN1'
date = '710'
probs1 = [0.1, 0.2, 0.4, 0.4, 0.1, 0.1, 0.2, 0.2, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
probs = [0.5, 0.6, 0.8, 0.8, 0.5, 0.5, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

from sklearn.model_selection import KFold
from itertools import islice
file_all = train_set['image_name'].values
test_file_all = test_set['image_name'].values
y_all = utils.get_y(train_set['tags'].values, label_map)
pred_tr = np.zeros((file_all.shape[0], 17))
pred_ts = np.zeros((test_file_all.shape[0], 17))
kf = KFold(n_splits = n_splits)

k_now = 0
for i_tr, i_vl in islice(kf.split(y_all), 0, None):
    model = core.MyNet(17).cuda()
    criterion = nn.BCELoss(weight=None).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    estimator = core.Estimator(model, criterion, optimizer)
    best_model = core.BestModel()
    
    for epoch in range(num_epoches):
        time_st = time()
        # 训练
        train_loader = utils.weighted_train_loader(file_all[i_tr], y_all[i_tr], probs, transform_tr, batch_size, pic_size)
        loss_tr = estimator.train(train_loader)

        # 验证
        val_loader = utils.valid_loader(file_all[i_vl], y_all[i_vl], transform_vl, batch_size, pic_size)
        loss_vl, f2_vl = estimator.validate(val_loader)
        best_model.update(loss_vl.avg, f2_vl, estimator.model)
        
        # 若验证结果提升缓慢，减小学习率
        if epoch > lr_tolerance and best_model.lrcount > lr_tolerance:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
            print('New Learn Rate: {}!'.format(optimizer.param_groups[0]['lr']))
            best_model.lrcount = 0
        
        # 若验证结果不再提升，保存模型、验证结果、预测结果，跳出迭代
        if epoch > tolerance and best_model.nobetter > tolerance:
            print('Early Stop in Epoch: {}, Best Val-Loss: {:.6f}, Best F2: {:.6f}'.format(
                epoch+1, best_model.best_loss, best_model.best_f2.value(bestf2=True)))
            pred_tr[i_vl, :] = f2_vl.preds
            best_model.save('./model{}-date{}-kf{}.pth'.format(model_name, date, k_now+1))
                                                               
            tst_loader = utils.test_loader(test_file_all, transform_vl, batch_size, pic_size)
            estimator.model.load_state_dict(best_model.best_model)                                       
            pred_ts_temp = estimator.predict(tst_loader)
            pred_ts =  pred_ts_temp / float(n_splits)
            np.save('./model{}_date{}_pred_train_kf{}.npy'.format(model_name, date, k_now+1), pred_tr)
            np.save('./model{}_date{}_pred_test_kf{}.npy'.format(model_name, date, k_now+1), pred_ts_temp)
            break
        
        # 打印每一次迭代的训练验证成绩
        print('{} [{}/{}] {}s, Loss: {:.4f}, Val-Loss: {:.4f}, Best Val-Loss: {:.4f}, Val-F2: {:.2f}'.format(
            k_now+1, epoch+1, num_epoches, int(time() - time_st), loss_tr.avg, loss_vl.avg, 
            best_model.best_loss, f2_vl.value(0.3)))
        gc.collect()
    k_now += 1

# 序列化验证和预测结果，用于stacking
np.save('./model{}_date{}_pred_train.npy'.format(model_name, date), pred_tr)
np.save('./model{}_date{}_pred_test.npy'.format(model_name, date), pred_ts)

th1 = utils.f2_opti_score(y_all, pred_tr, thresholds = np.arange(0, 1, 0.01), num_classes=17)
th2 = utils.f2_opti_score(y_all, pred_tr, thresholds = np.arange(1, 0, -0.01), num_classes=17)
th = (th1 + th2) / 2.0
print(utils.f2_score(y_all, pred_tr, th))
submit_df = utils.to_submit(pred_ts, th, test_set, inv_label_map)
submit_df.to_csv('./submit/model{}_date{}_no{}.csv'.format(model_name, date, 1), index=False)