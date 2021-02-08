
from collections import OrderedDict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchsummary import summary




torch.cuda.empty_cache()
date = ''
LR=0.001
weight_decay=0.0001
BATCH_SIZE=8
EPOCHS=160
train_r2 = 0.2460
test_r2 = 0.2520

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:1')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
else:
    DEVICE = torch.device('cpu')

train_data_set = []
train_modeloutput_set = []
train_loss_set = []
test_data_set = []
test_modeloutput_set = []
test_loss_set = []
R2_score_set = []
RMSE_set = []
R2_score_train_set = []

sys.stdout = Logger('20201015.log', sys.stdout)
sys.stderr = Logger('20201015.log_file', sys.stderr)		# redirect std err, if necessary
    
train_dataset = PermEstimateDataset(ext='data')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, # 分批次训练
                          shuffle=True,  num_workers=int(0))

test_dataset = testPermEstimateDataset(ext='data')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=int(0))

model = DenseNet().to(DEVICE)
# model.apply(weights_init)

if weight_decay>0:
   reg_loss=Regularization(model, weight_decay, p=2).to(DEVICE)   #p=1为L1正则化，P=2为L2正则化
else:
   print("no regularization")
optimizer = optim.Adam(model.parameters(),lr=LR,weight_decay=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[80,120],gamma = 0.2)

print("------------device:{}------------------".format(DEVICE))
print("------------Pytorch version:{}---------------".format(torch.__version__))

torch.cuda.synchronize()
start = time.time()


for epoch in range(1,EPOCHS +1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader, epoch)
    scheduler.step()  # 更新学习率

torch.cuda.synchronize()
end = time.time()
time_elapsed = end - start
print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)) 

torch.save(model.state_dict(), '20201015.pkl')

plotandsave

