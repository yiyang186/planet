import torch
from torch import nn

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
    
class MyNetfirst4(nn.Module):
    def __init__(self, num_classes=17):
        super(MyNetfirst4, self).__init__()
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
            nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
        x = self.classifier(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, dropout=0.25):
        padding = int(kernel_size / 2)
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
            nn.Dropout(dropout),     
        )
        self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out += self.downsample(x)
        out = self.relu(out)
        return out

class ProposedBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ProposedBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),   
        )
        self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv(x)
        if self.stride > 1:
            out += self.downsample(x)
        else:
            out += x
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=17, block=BasicBlock):
        super(ResNet, self).__init__()

        self.conv = nn.Sequential(
            block(3, 32, kernel_size=5, dropout=0.25),
            block(32, 64, kernel_size=3, dropout=0.25),
            block(64, 128, kernel_size=3, dropout=0.5),
            block(128, 256, kernel_size=3, dropout=0.5),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x