from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# from dataloader import Dataset
from dataloader_scanobject import ScanObjectNN
from dataloader_modelnet40 import Dataset_m40
from model import  DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from termcolor import colored
import os
import copy
import seaborn as sns
import pickle
import random
from tqdm import tqdm
pt = 'lth'
pruning_round =0
# Custom Libraries
from utils import get_split, compare_models, checkdir, print_nonzeros
from weights import weight_init, weight_rewinding
from prune import prune_rate_calculator, lth_local_unstructured_pruning, lth_global_unstructured_pruning


from dataloader_shapenetcore import Dataset_shapenet

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='shapenetcore', metavar='N',                           # try
                        choices=['scanobjectnn', 'modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,                                                  # try
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.4,                               # try   previous 0.5
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()
    





# #root = 'G:\ModelNet40 Dataset'  # <<<<<<< select root 
# root = '/home/ece-desm/Ismail/ICPR_24/ModelNet40_Dataset'                                                     # try for modelnet40
# dataset_name = 'modelnet40'

# root= '/home/ece-desm/Ismail/ICPR_24/ShapeNetCore_Dataset'                                                         # try for shapenetore
# dataset_name = 'shapenetcorev2'
batch_size= 32
test_batch_size= 128
workers= 16

# train_loader = DataLoader(Dataset(root=root, dataset_name=dataset_name, num_points=2048, split='trainval'), num_workers=8,
#                               batch_size= batch_size, shuffle=True, drop_last=True)
# test_loader = DataLoader(Dataset(root=root, dataset_name=dataset_name, num_points=2048, split='test'), num_workers=8,
#                              batch_size= test_batch_size, shuffle=True, drop_last=False)

if(args.dataset=='scanobjectnn'):
    dataset = ScanObjectNN(2500)
    test_dataset = ScanObjectNN(2500, 'test')

    print("train size: ",dataset.__len__())
    print("test size: ",test_dataset.__len__())

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size,num_workers=workers, pin_memory=True, shuffle=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("model is running in:  ", device)

    model = DGCNN(args,output_channels=15).to(device)


elif(args.dataset=='modelnet40'):
    #root = '/home/ece-desm/Ismail/ICPR_24/ModelNet40_Dataset'                                 # Rtx 4090
    # root= '/home/xavier/research/Amrijit/ModelNet40'                                            # Rtx 5000
    root = '/shafinSSD/Amrijit/ModelNet40_Dataset'
    dataset_name = 'modelnet40'

    dataset = Dataset_m40(root=root, dataset_name=dataset_name, num_points=1024, split='trainval')
    test_dataset = Dataset_m40(root=root, dataset_name=dataset_name, num_points=1024, split='test')

    print("train size: ",dataset.__len__())
    print("test size: ",test_dataset.__len__())

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size,num_workers=workers, pin_memory=True, shuffle=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("model is running in:  ", device)

    model = DGCNN(args,output_channels=40).to(device)

elif(args.dataset=='shapenetcore'):
    #root = 'G:\ShapeNetCoreV2'                                         # Amrijit laptop
    root = '/shafinSSD/Amrijit/ShapeNetCoreV2_Dataset'        # Rtx4090
    dataset_name = 'shapenetcorev2'

    dataset = Dataset_shapenet(root=root, dataset_name=dataset_name, num_points=2048, split='trainval')
    test_dataset = Dataset_shapenet(root=root, dataset_name=dataset_name, num_points=2048, split='test')

    print("train size: ",dataset.__len__())
    print("test size: ",test_dataset.__len__())

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers= workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers= workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("model is running in:  ", device)

    model = DGCNN(args,output_channels=55).to(device)

else:
     print("invalid dataset!!")




opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
criterion = cal_loss


prune_rate = 0
final_prune_rate = 99
bias = False
bestacc1 = []

#checkdir(f"{os.getcwd()}/saves/pointnet/modelnet40")
#weight_dir = f"{os.getcwd()}/saves/pointnet/modelnet40/init_for_lth.pth"
model0 = DGCNN(args,output_channels=55).to(device)
#model0 = PointNetCls(k=num_classes, feature_transform=feature_transform).to(device=device)
#if os.path.exists(weight_dir):
#initial_state_dict = torch.load(weight_dir)
#model0.load_state_dict(initial_state_dict)
#else:

initial_state_dict = copy.deepcopy(model0.state_dict())
#checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
#torch.save(model0.state_dict(), weight_dir)
model.load_state_dict(initial_state_dict)
model.to(device)
best_model = model
# Optimizer and Scheduler

for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            # loss = criterion(logits, label)




opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)

opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
criterion = cal_loss
best_test_accuracy2 = 0.0 
num_classes = 55
while (prune_rate < final_prune_rate):
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    checkdir(f"{os.getcwd()}/saves/dgcnn/{args.dataset}/full")
    # checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/B")
    torch.save(model.state_dict(),f"{os.getcwd()}/saves/dgcnn/{args.dataset}/full/{prune_rate:.2f}_model_{pt}.pth")
    #optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    if not pruning_round == 0:
        print(pruning_round)
        model = lth_local_unstructured_pruning(model,num_classes, bias=bias, conv=0.20, linear=0.20, output=0, batchnorm=0)#lth_local_unstructured_pruning(classifier,num_classes, bias=bias, conv=0.99, linear=0.99, output=0, batchnorm=0)#lth_global_unstructured_pruning(classifier, num_classes, bias=bias, conv=0.99, linear=0.99, output=0, batchnorm=0)lth_local_unstructured_pruning(classifier,num_classes, bias=bias, conv=0.10, linear=0.10, output=0, batchnorm=0)
        #model = lth_global_unstructured_pruning(model, num_classes, bias=bias, conv=0.60, linear=0.60, output=0.0 ,batchnorm=0)
        model = weight_rewinding(model, model0, bias=bias)
        prune_rate = prune_rate_calculator(model, bias=bias)
    
    best_test_accuracy = 0.0  # Initialize the best accuracy variable
    #best_epoch = 0
    n_epochs = 80
    best_epoch = 0
    
    
    for epoch in tqdm(range(args.epochs)):
            #scheduler.step()
            ####################
            # Train
            ####################
            train_loss = 0.0
            count = 0.0
            model.train()
            train_pred = []
            train_true = []
            for data, label in train_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()
                logits = model(data)
                loss = criterion(logits, label)
                loss.backward()
                opt.step()
                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
            scheduler.step()
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (epoch,
                                                                train_loss*1.0/count,
                                                                metrics.accuracy_score(
                                                                train_true, train_pred))
            print(outstr)

            ###################
            ###    Test
            ###################
            test_loss = 0.0
            count = 0.0
            best_test_acc=0 
            model.eval()
            test_pred = []
            test_true = []
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f' % (epoch,
                                                              test_loss*1.0/count,
                                                              test_acc)
            print(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                checkpoint_dir = 'checkpoints'
                #best_test_accuracy = test_acc
            if best_test_accuracy <test_acc:
                best_test_accuracy = test_acc
            
            # Create directory if it doesn't exist
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Save model state_dict
            checkpoint_path = os.path.join(checkpoint_dir, 'model.t7')
            torch.save(model.state_dict(), checkpoint_path)
    bestacc1.append(best_test_accuracy)
    if best_test_accuracy2<best_test_accuracy:
            best_test_accuracy2 = best_test_accuracy
    pruning_round += 1

    #classifier = best_model
    print(colored('Test Accuracy: {:.3f} | Best Test Accuracy: {:.3f}'.format(best_test_accuracy,best_test_accuracy2), 'yellow'))

checkdir(f"{os.getcwd()}/dumps/lth/dgcnn/{args.dataset}/local/full")
import numpy as np
np_bestacc1 = np.array([x for x in bestacc1])
np_bestacc1.dump(f"{os.getcwd()}/dumps/lth/dgcnn/{args.dataset}/local/full/all_accuracy.dat")  




def test(args):
    # test_loader = DataLoader(Dataset(root=root, dataset_name=dataset_name, num_points=1024, split='test'), num_workers=0,
    #                          batch_size= test_batch_size, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
                                test_dataset, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    checkpoint_dir= 'checkpoints/model.t7'
    model.load_state_dict(torch.load(checkpoint_dir))
    #model.load_state_dict(torch.load('checkpoints/dgcnn_1024/models/model.t7')) 
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    print(outstr)


if args.eval:
     test(args)
else:
     print("Training the model-----------------")
