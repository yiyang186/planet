import torch
import utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt 

class BestModel(object):
    def __init__(self):
        self.best_loss = 10000
        self.best_f2 = None
        self.best_model = None
        self.nobetter = 0
        self.lrcount = 0
    
    def update(self, loss, f2, model):
        if abs(loss) < abs(self.best_loss):
            self.best_loss = loss
            self.best_f2 = f2
            print("Update Model!")
            self.lrcount = self.nobetter = 0
            self.best_model = model.state_dict()
        else:
            self.lrcount += 1
            self.nobetter += 1
    
    def save(self, path):
        torch.save(self.best_model, path)

class F2Meter(object):
    def __init__(self, num_classes):
        self.best_th = None
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.labels = np.array([]).reshape(0, self.num_classes)
        self.preds = np.array([]).reshape(0, self.num_classes)
        
    def update(self, label, out):
        self.labels = np.concatenate((self.labels, label.data.cpu().numpy()))
        self.preds = np.concatenate((self.preds, out.data.cpu().numpy()))
        
    def value(self, th=0.3, bestf2=False):
        labels, preds = np.array(self.labels), np.array(self.preds)
        if not bestf2:
            return utils.f2_score(labels, preds, np.zeros(self.num_classes)+th, num_classes=self.num_classes)
        else:
            th1 = utils.f2_opti_score(labels, preds, thresholds = np.arange(0, 1, 0.01), num_classes=self.num_classes)
            th2 = utils.f2_opti_score(labels, preds, thresholds = np.arange(1, 0, -0.01), num_classes=self.num_classes)
            self.best_th = (th1+th2)/2
            return utils.f2_score(labels, preds, self.best_th, num_classes=self.num_classes)
    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def output(self):
        return self.avg

class Estimator(object):
    def __init__(self, model, criterion, optimizer, num_classes):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_classes = num_classes

    def train(self, train_loader):
        loss_tr = AverageMeter()
        self.model.train()
        for img, label in train_loader:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            # 向前传播
            out = self.model(img)
            loss = self.criterion(out, label.float()) # for BCELoss
            # loss = self.criterion(out, label.long()) # for CrossEntropyLoss
            loss_tr.update(loss.data[0], label.size(0))

            # 向后传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_tr
            
    def validate(self, val_loader):
        loss_vl = AverageMeter()
        f2_vl = F2Meter(self.num_classes)
        self.model.eval()
        for img, label in val_loader:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()

            out = self.model(img)
            loss = self.criterion(out, label.float()) # for BCELoss
            # loss = self.criterion(out, label.long()) # for CrossEntropyLoss
            loss_vl.update(loss.data[0], label.size(0))
            f2_vl.update(label, out)
        return loss_vl, f2_vl
    
    def predict(self, tst_loader):
        self.model.eval()
        preds = np.array([]).reshape(0, self.num_classes)
        for data in tst_loader:
            img = Variable(data, volatile=True).cuda()
            out = self.model(img)
            preds = np.concatenate((preds, out.data.cpu().numpy()))
        return preds

class History(object):
    def __init__(self, names):
        self.data = {}
        for n in names:
            self.data[n] = np.array([], dtype='float')

    def update(self, d):
        for (k, v) in d.items():
            self.data[k] = np.append(self.data[k], v)

    def plot(self, names):
        plt.figure(figsize=(10, 8))
        for n in names:
            plt.plot(self.data[n], label=n)
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()
