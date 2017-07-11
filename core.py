import torch
import utils
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
    def __init__(self):
        self.reset()
        self.best_th = None

    def reset(self):
        self.labels = np.array([]).reshape(0, 17)
        self.preds = np.array([]).reshape(0, 17)
        
    def update(self, label, out):
        self.labels = np.concatenate((self.labels, label.data.cpu().numpy()))
        self.preds = np.concatenate((self.preds, out.data.cpu().numpy()))
        
    def value(self, th=0.3, bestf2=False):
        labels, preds = np.array(self.labels), np.array(self.preds)
        if not bestf2:
            return utils.f2_score(labels, preds, np.zeros(17)+th)
        else:
            th1 = utils.f2_opti_score(labels, preds, thresholds = np.arange(0, 1, 0.01), num_classes=17)
            th2 = utils.f2_opti_score(labels, preds, thresholds = np.arange(1, 0, -0.01), num_classes=17)
            self.best_th = (th1+th2)/2
            return utils.f2_score(labels, preds, self.best_th)
    
    
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
        
class MyNet(nn.Module):
    def __init__(self, num_classes=17):
        super(MyNet, self).__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Linear(4*4*256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
        x = self.classifier(x)
        return x

class Estimator(object):
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, train_loader):
        loss_tr = AverageMeter()
        self.model.train()
        for img, label in train_loader:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            # 向前传播
            out = self.model(img)
            loss = self.criterion(out, label.float())
            loss_tr.update(loss.data[0], label.size(0))

            # 向后传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_tr
            
    def validate(self, val_loader):
        loss_vl = AverageMeter()
        f2_vl = F2Meter()
        self.model.eval()
        for img, label in val_loader:
            img = Variable(img, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()

            out = self.model(img)
            loss = self.criterion(out, label.float())
            loss_vl.update(loss.data[0], label.size(0))
            f2_vl.update(label, out)
        return loss_vl, f2_vl
    
    def predict(self, tst_loader):
        self.model.eval()
        preds = np.array([]).reshape(0, 17)
        for data in tst_loader:
            img = Variable(data, volatile=True).cuda()
            out = self.model(img)
            preds = np.concatenate((preds, out.data.cpu().numpy()))
        return preds