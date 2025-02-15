import argparse 
import math
import h5py
import numpy as np
import socket
import importlib
import matplotlib.pyplot as plt
import os
import sys
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
from dataloader import Dataset


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
        
NUM_CLASS = 55
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

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
loss_fn = nn.CrossEntropyLoss()

global_step = 1

#model_save_dir = os.path.join(CURRENT_DIR, "models", "mnist2")
#os.makedirs(model_save_dir, exist_ok = True)

# TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/shapenetcorev2_ply_hdf5_2048/train_files.txt'))        # try
# TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/shapenetcorev2_ply_hdf5_2048/test_files.txt'))
# /home/ece-desm/Ismail/ICPR_24/Models/PointCNN.Pytorch/data/shapenetcorev2_ply_hdf5_2048
TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))        # try
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

# print(TRAIN_FILES)


# # Dataset
# root= '/shafinSSD/Amrijit/BMVC_Models/PointCNN_ShapeNetCore/data'
# split_train = 'trainval'
# split_test = 'test'
# dataset_name = 'shapenetcorev2'
# batch_size= 32
# workers= 8

# dataset = Dataset(root=root, dataset_name=dataset_name, num_points=2048, split='trainval')
# test_dataset = Dataset(root=root, dataset_name=dataset_name, num_points=2048, split='test')

# # Data loaders
# dataloader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

# testdataloader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)




# # Training Loop
# for epoch in range(NUM_EPOCHS):
#     model.train()  # Set the model to training mode
#     total_loss = 0.0

#     for batch_idx, (data, target) in enumerate(dataloader):
#         data, target = data.cuda(), target.cuda()  # Move data to GPU
#         optimizer.zero_grad()  # Zero the gradients

#         # Forward pass
#         output = model(data)
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
#                 output = model(data)
#                 test_loss += loss_fn(output, target).item()
#                 pred = output.argmax(dim=1, keepdim=True)
#                 correct += pred.eq(target.view_as(pred)).sum().item()

#         test_loss /= len(testdataloader.dataset)
#         accuracy = 100. * correct / len(testdataloader.dataset)
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#             test_loss, correct, len(testdataloader.dataset), accuracy))





losses = []
accuracies = []

'''
if False:
    latest_model = sorted(os.listdir(model_save_dir))[-1]
    model.load_state_dict(torch.load(os.path.join(model_save_dir, latest_model)))    

'''


best_accuracy = 0.0  # Initialize the best accuracy variable

for epoch in range(1, NUM_EPOCHS+1):
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(TRAIN_FILES)):
        #log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:, 0:NUM_POINT, :]

        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        if epoch > 1:
            LEARNING_RATE *= DECAY_STEP ** (global_step // DECAY_STEP)
            if LEARNING_RATE > LEARNING_RATE_MIN:
                print("NEW LEARNING RATE:", LEARNING_RATE)
                optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            # Label
            label = current_label[start_idx:end_idx]
            label = torch.from_numpy(label).long()
            label = Variable(label, requires_grad=False).cuda()
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)  # P_Sampled
            P_sampled = jittered_data
            F_sampled = np.zeros((BATCH_SIZE, NUM_POINT, 0))
            optimizer.zero_grad()

            t0 = time.time()
            P_sampled = torch.from_numpy(P_sampled).float()
            P_sampled = Variable(P_sampled, requires_grad=False).cuda()

            out = model((P_sampled, P_sampled))
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            pred_labels = torch.argmax(out, dim=1)
            correct = (pred_labels == label).sum().item()
            total_correct += correct
            total_seen += BATCH_SIZE

            accuracy = total_correct / total_seen
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), 'best_model_weights.pth')

            # print("epoch: {}, batch: {}, loss: {:.4f}, accuracy: {:.2f}%".format(
            #     epoch, batch_idx, loss.item(), correct / BATCH_SIZE * 100))
            # print("best accuracy >>> ", best_accuracy * 100)
            global_step += 1
# print("Epoch: {}, Loss: {:.4f}, Accuracy: {:.2f}%".format(epoch, loss.item(), total_correct / total_seen * 100))
# print("Best accuracy so far: {:.2f}%".format(best_accuracy * 100))
    avg_train_accuracy = total_correct / total_seen * 100

    # Testing
    test_accuracy = 0.0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for test_file in TEST_FILES:
            current_data, current_label = provider.loadDataFile(test_file)
            current_data = current_data[:, 0:NUM_POINT, :]
            current_label = np.squeeze(current_label)

            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE

            total_correct = 0
            total_seen = 0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE

                label = current_label[start_idx:end_idx]
                label = torch.from_numpy(label).long()
                label = Variable(label, requires_grad=False).cuda()

                P_sampled = torch.from_numpy(current_data[start_idx:end_idx]).float()
                P_sampled = Variable(P_sampled, requires_grad=False).cuda()

                out = model((P_sampled, P_sampled))
                pred_labels = torch.argmax(out, dim=1)
                correct = (pred_labels == label).sum().item()
                total_correct += correct
                total_seen += BATCH_SIZE

            if total_seen > 0:
                test_accuracy += total_correct / total_seen * 100

    test_accuracy /= len(TEST_FILES) if len(TEST_FILES) > 0 else 1
    print("Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%".format(
        epoch, loss.item(), avg_train_accuracy, test_accuracy))
    model.train()  # Set the model back to training mode

