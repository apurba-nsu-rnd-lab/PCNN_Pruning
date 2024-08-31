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
from termcolor import colored
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import provider
import math
import random
#from utils import data_utils
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


from tqdm import tqdm
pt = 'lth'
pruning_round =0
# Custom Libraries
from util import get_split, compare_models, checkdir, print_nonzeros
from weights import weight_init, weight_rewinding
from prune import prune_rate_calculator, lth_local_unstructured_pruning, lth_global_unstructured_pruning
# def lth_local_unstructured_pruning(model, output_class, bias=True, **layers):


prune_rate = 0
final_prune_rate = 96
bias = False
bestacc1 = []

random.seed(0)
dtype = torch.cuda.FloatTensor



# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=30, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--dataset', type=str, default= 'shapenetcore', choices=['shapenetcore', 'modelnet40', 'scanobjectnn'])
# parser.add_argument('--target_class', type=int, default=55, choices=[55,40,15])
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
#NUM_CLASS= FLAGS.target_class

if (FLAGS.dataset=='shapenetcore'):
    NUM_CLASS = 55
elif(FLAGS.dataset=='modelnet40'):
    NUM_CLASS = 40
elif(FLAGS.dataset=='scanobjectnn'):
    NUM_CLASS = 15
else:
    print("invalid dataset !!!")


BATCH_SIZE = FLAGS.batch_size #32
NUM_EPOCHS = FLAGS.max_epoch
jitter = 0.01
jitter_val = 0.01

rotation_range = [0, math.pi / 18, 0, 'g']
rotation_rage_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']



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
    root= '/shafinSSD/Amrijit/BMVC_Models/PointCNN_ShapeNetCore/data'                     # Rtx3090
    #root= '/home/ece-desm/Ismail/ICPR_24/ShapeNetCore_Dataset'                              # Rtx 4090
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
    root = '/shafinSSD/Amrijit/ModelNet40_Dataset'                            # Rtx 3090
    # root = '/home/ece-desm/Ismail/ICPR_24/ModelNet40_Dataset'                   # Rtx 4090
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
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model0 = Classifier().cuda()
initial_state_dict = copy.deepcopy(model0.state_dict())
#checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
#torch.save(model0.state_dict(), weight_dir)
model.load_state_dict(initial_state_dict)
model.to(device)
best_model = model

best_test_accuracy2 = 0.0 
num_classes = NUM_CLASS

import copy
pruning_round = 0

while (prune_rate < final_prune_rate):
    opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, NUM_EPOCHS, eta_min=0.00001)
    
    checkdir(f"{os.getcwd()}/saves/pointcnn/{FLAGS.dataset}/full")
    # checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/B")
    torch.save(model.state_dict(),f"{os.getcwd()}/saves/pointcnn/{FLAGS.dataset}/full/{prune_rate:.2f}_model{pt}.pth")
    #optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    if not pruning_round == 0:
        print(pruning_round)
        model = lth_local_unstructured_pruning(model,num_classes, bias=bias, conv=0.20, linear=0.20, output=0, batchnorm=0)#lth_local_unstructured_pruning(classifier,num_classes, bias=bias, conv=0.99, linear=0.99, output=0, batchnorm=0)#lth_global_unstructured_pruning(classifier, num_classes, bias=bias, conv=0.99, linear=0.99, output=0, batchnorm=0)lth_local_unstructured_pruning(classifier,num_classes, bias=bias, conv=0.10, linear=0.10, output=0, batchnorm=0)
        #model = lth_global_unstructured_pruning(model, num_classes, bias=bias, conv=0.20, linear=0.20, output=0.0 ,batchnorm=0)
        model = weight_rewinding(model, model0, bias=bias)
        prune_rate = prune_rate_calculator(model, bias=bias)
    
    best_test_accuracy = 0.0  # Initialize the best accuracy variable
    #best_epoch = 0
    n_epochs = 80
    best_epoch = 0

   
    print(prune_rate)
    
    print("\nModel's Parameters:")


    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0





    for epoch in tqdm(range(NUM_EPOCHS)):
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

            
        scheduler.step()
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
            best_test_accuracy  = accuracy
            best_accuracy = accuracy
            best_model_weights = copy.deepcopy(model.state_dict())

    # After training, load the best model weights for testing
    model.load_state_dict(best_model_weights)
    bestacc1.append(best_test_accuracy)
    if best_test_accuracy2<best_test_accuracy:
            best_test_accuracy2 = best_test_accuracy
    pruning_round += 1

    #classifier = best_model
    print(colored('Test Accuracy: {:.3f} | Best Test Accuracy: {:.3f}'.format(best_test_accuracy,best_test_accuracy2), 'yellow'))

checkdir(f"{os.getcwd()}/dumps/lth/pointcnn/{FLAGS.dataset}/local/full")
import numpy as np
np_bestacc1 = np.array([x for x in bestacc1])
np_bestacc1.dump(f"{os.getcwd()}/dumps/lth/pointcnn/{FLAGS.dataset}/local/full/all_accuracy.dat")  

