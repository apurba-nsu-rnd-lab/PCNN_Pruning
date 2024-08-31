import argparse 
import math
import h5py
import numpy as np
import socket
import importlib
import matplotlib.pyplot as plt
import os
import sys
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import provider
import math
import random
from utils import data_utils
import time

import torch
from torch import nn
from torch.autograd import Variable
#from torch.utils.data import Dataset, DataLoader
from dataloader_modelnet40 import Modelnet_Dataset
from dataloader_scanobjectnn import ScanObjectNN
from dataloader_shapenetcore import Shapenet_Dataset


from utils.model import RandPointCNN
from utils.util_funcs import knn_indices_func_gpu, knn_indices_func_cpu
from utils.util_layers import Dense


random.seed(0)
dtype = torch.cuda.FloatTensor



# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--dataset', type=str, default= 'shapenetcore', choices=['shapenetcore', 'modelnet40', 'scanobjectnn'])
parser.add_argument('--target_class', type=int, default=55, choices=[55,40,15])
FLAGS = parser.parse_args()

NUM_POINT = FLAGS.num_point
LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
       
MAX_NUM_POINT = 2048

DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


LEARNING_RATE_MIN = 0.00001

# select no of classes via dataset name
NUM_CLASS= FLAGS.target_class

# if (FLAGS.dataset=='shapenetcore'):
#     NUM_CLASS = 55
# elif(FLAGS.dataset=='modelnet40'):
#     NUM_CLASS = 40
# elif(FLAGS.dataset=='scanobjectnn'):
#     NUM_CLASS = 15
# else:
#     print("invalid dataset !!!")


BATCH_SIZE = FLAGS.batch_size #32
NUM_EPOCHS = FLAGS.max_epoch
jitter = 0.01
jitter_val = 0.01

rotation_range = [0, math.pi / 18, 0, 'g']
rotation_rage_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']


# class modelnet40_dataset(Dataset):

#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         return self.data[i], self.labels[i]


# C_in, C_out, D, N_neighbors, dilution, N_rep, r_indices_func, C_lifted = None, mlp_width = 2
# (a, b, c, d, e) == (C_in, C_out, N_neighbors, dilution, N_rep)
# Abbreviated PointCNN constructor.
AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, NUM_CLASS, with_bn=False, activation=None)
        )

    def forward(self, x):
        x = self.pcnn1(x)
        if False:
            print("Making graph...")
            k = make_dot(x[1])

            print("Viewing...")
            k.view()
            print("DONE")

            assert False
        x = self.pcnn2(x)[1]  # grab features

        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean


print("------Building model-------")
model = Classifier().cuda()
print("------Successfully Built model-------")

# print(model)
import torch.optim as optim
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9)
loss_fn = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
global_step = 1

#   >>>>>>>>>>>>>>>>>>>> selecting dataset and dataloader based on input <<<<<<<<<<<<<<<<<<<<<<
if (FLAGS.dataset=='shapenetcore'):
    # Dataset
    root= '/shafinSSD/Amrijit/BMVC_Models/PointCNN_ShapeNetCore/data'
    split_train = 'trainval'
    split_test = 'test'
    dataset_name = 'shapenetcorev2'
    batch_size= 32
    workers= 16

    dataset = Shapenet_Dataset(root=root, dataset_name=dataset_name, num_points=2048, split='trainval')
    test_dataset = Shapenet_Dataset(root=root, dataset_name=dataset_name, num_points=2048, split='test')
    print("datasize:", dataset.__len__())
    print("datasize:", test_dataset.__len__())

    # Data loaders
    dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    testdataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

elif(FLAGS.dataset=='modelnet40'):
    #root = 'G:\ModelNet40 Dataset'  # <<<<<<< select root 
    root = '/shafinSSD/Amrijit/ModelNet40_Dataset'
    dataset_name = 'modelnet40'
    batch_size= 128
    test_batch_size= 32
    workers= 16

    dataset = Modelnet_Dataset(root=root, dataset_name=dataset_name, num_points=1024, split='trainval')
    test_dataset = Modelnet_Dataset(root=root, dataset_name=dataset_name, num_points=1024, split='test')
    print("datasize:", dataset.__len__())
    print("datasize:", test_dataset.__len__())

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=workers,
                              batch_size= batch_size, shuffle=True, drop_last=True)
    
    testdataloader = torch.utils.data.DataLoader(test_dataset, num_workers=workers,
                             batch_size= test_batch_size, shuffle=True, drop_last=False)

elif(FLAGS.dataset=='scanobjectnn'):
    batch_size= 32
    test_batch_size= 32
    workers= 16
    # Dataset
    dataset = ScanObjectNN(2500)
    test_dataset = ScanObjectNN(2500, 'test')

    print("train size: ",dataset.__len__())
    print("test size: ",test_dataset.__len__())


# Data loaders
    dataloader = torch.utils.data.DataLoader(
                 dataset, batch_size=batch_size,num_workers=workers, pin_memory=True, shuffle=True, drop_last=True)

    testdataloader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=False)

else:
    print("invalid dataset !!!")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Training loop start <<<<<<<<<<<<<<<<<<<<<<<<<



import copy

best_model_weights = copy.deepcopy(model.state_dict())
best_accuracy = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()  # Move data to GPU
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        output = model((data, data))  # Pass both points and features
        target = target.view(-1)  # Reshape the target tensor

        loss = loss_fn(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 99:  # Print every 100 mini-batches
            print('[%d, %5d] Train loss: %.3f' %
                  (epoch + 1, batch_idx + 1, total_loss / 100))
            total_loss = 0.0

    # Validation
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testdataloader:
            data, target = data.cuda(), target.cuda()
            output = model((data, data))
            target = target.view(-1)

            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testdataloader.dataset)
    accuracy = 100. * correct / len(testdataloader.dataset)
    print('Epoch: {}, Test Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch + 1, test_loss, correct, len(testdataloader.dataset), accuracy))

    # Save the best model weights
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_weights = copy.deepcopy(model.state_dict())

# After training, load the best model weights for testing
model.load_state_dict(best_model_weights)





# import copy

# best_model_weights = copy.deepcopy(model.state_dict())
# best_accuracy = 0.0
# NUM_EPOCHS = 50
# for epoch in range(NUM_EPOCHS):
#     model.train()  # Set the model to training mode
#     total_loss = 0.0

#     for batch_idx, (data, target) in enumerate(dataloader):
#         t1 = time.time()
#         data, target = data.cuda(), target.cuda()  # Move data to GPU
#         optimizer.zero_grad()  # Zero the gradients
#         t3 = time.time()
#         #print('data loading time',t3-t1)
#         # Forward pass
#         output = model((data, data))  # Pass both points and features
#         target = target.view(-1)  # Reshape the target tensor
#         t3 = time.time()
#         #print('data loading time',t3-t1)
#         loss = loss_fn(output, target)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()
#         t2 = time.time()
#         #print('total time, ',t2-t1)

#         total_loss += loss.item()

#         if batch_idx % 100 == 99:  # Print every 100 mini-batches
#             print('[%d, %5d] Train loss: %.3f' %
#                   (epoch + 1, batch_idx + 1, total_loss / 100))
#             total_loss = 0.0
#     scheduler.step()

#     # Validation
#     model.eval()  # Set the model to evaluation mode
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in testdataloader:
#             data, target = data.cuda(), target.cuda()
#             output = model((data, data))
#             target = target.view(-1)

#             test_loss += loss_fn(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(testdataloader.dataset)
#     accuracy = 100. * correct / len(testdataloader.dataset)
#     print('Epoch: {}, Test Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
#         epoch + 1, test_loss, correct, len(testdataloader.dataset), accuracy))

#     # Save the best model weights
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_model_weights = copy.deepcopy(model.state_dict())

# # After training, load the best model weights for testing
# model.load_state_dict(best_model_weights)




# # Training Loop
# for epoch in range(NUM_EPOCHS):
#     model.train()  # Set the model to training mode
#     total_loss = 0.0

#     for batch_idx, (data, target) in enumerate(dataloader):
#         data, target = data.cuda(), target.cuda()  # Move data to GPU
#         optimizer.zero_grad()  # Zero the gradients

#         # Forward pass
#         output = model((data, data))  # Pass both points and features
#         target = target.view(-1)      #  or target = target.squeeze()
#         loss = loss_fn(output, target)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         if batch_idx % 100 == 99:  # Print every 100 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, batch_idx + 1, total_loss / 100))
#             total_loss = 0.0

#     # Validation
#     if epoch % 5 == 0:
#         model.eval()  # Set the model to evaluation mode
#         test_loss = 0
#         correct = 0
#         with torch.no_grad():
#             for data, target in testdataloader:
#                 data, target = data.cuda(), target.cuda()
#                 output = model((data, data))
#                 target = target.view(-1)
#                 test_loss += loss_fn(output, target).item()
#                 pred = output.argmax(dim=1, keepdim=True)
#                 correct += pred.eq(target.view_as(pred)).sum().item()

#         test_loss /= len(testdataloader.dataset)
#         accuracy = 100. * correct / len(testdataloader.dataset)
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#             test_loss, correct, len(testdataloader.dataset), accuracy))

