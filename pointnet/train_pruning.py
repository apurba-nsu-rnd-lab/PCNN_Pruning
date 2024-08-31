import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from dataloader_modelnet40 import Dataset_m40
from dataloader_ScanobjectNN import ScanObjectNN
from dataloader_shapenetcore import Dataset_shapenet
from torch.utils.data import DataLoader
from torchvision import transforms
from model import PointNetCls, feature_transform_regularizer
from termcolor import colored
import time


import os
import copy
import seaborn as sns
import pickle
import random
pt = 'lth'
pruning_round =0
# Custom Libraries
from utils import get_split, compare_models, checkdir, print_nonzeros
from weights import weight_init, weight_rewinding
from prune import colt, prune_rate_calculator, lth_local_unstructured_pruning, lth_global_unstructured_pruning



parser = argparse.ArgumentParser(description='Point Cloud Classification')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--dataset', type=str, default='shapenetcore',
                        choices=['modelnet40', 'scanobjectnn', 'shapenetcore'])
parser.add_argument('--folder', type=str, default='shape',
                        choices=['m40', 'scan', 'shape'])
args = parser.parse_args()

# Set random seed
random_seed = 42
torch.manual_seed(random_seed)

#dgcnn

# Hyperparameters
batch_size = 256           
n_epochs = 100
feature_transform = True
lr = 0.001
weight_decay = 1e-4

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############
if(args.dataset=='modelnet40'):
    dataset_name = 'modelnet40'
    #root = 'G:\ModelNet40 Dataset'  # <<<<<<< select root 
    root = '/shafinSSD/Amrijit/ModelNet40_Dataset'#'/home/ece-desm/Ismail/ICPR_24/ModelNet40_Dataset'
    dataloader = DataLoader(Dataset_m40(root=root, dataset_name=dataset_name, num_points=2048, split='trainval'), num_workers=args.workers,
                              batch_size= batch_size, shuffle=True, drop_last=True)
    testdataloader = DataLoader(Dataset_m40(root=root, dataset_name=dataset_name, num_points=2048, split='test'), num_workers=args.workers,
                             batch_size= batch_size, shuffle=True, drop_last=False)
    num_classes = 40
    classifier = PointNetCls(k=num_classes, feature_transform=feature_transform).to(device=device)
    
elif(args.dataset=='scanobjectnn'):
    # Dataset
    dataset = ScanObjectNN(2500)
    test_dataset = ScanObjectNN(2500, 'test')

    # Data loaders
    dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers)

    num_classes = 15
    classifier = PointNetCls(k=num_classes, feature_transform=feature_transform).to(device=device)

elif(args.dataset=='shapenetcore'):
    #root = 'G:\ShapeNetCoreV2'
    root = '/shafinSSD/Amrijit/ShapeNetCoreV2_Dataset'
    split_train = 'trainval'
    split_test = 'test'
    dataset_name = 'shapenetcorev2'

    dataset = Dataset_shapenet(root=root, dataset_name=dataset_name, num_points=2048, split='trainval')
    test_dataset = Dataset_shapenet(root=root, dataset_name=dataset_name, num_points=2048, split='test')

    # Data loaders
    dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers)

    num_classes = 55
    classifier = PointNetCls(k=num_classes, feature_transform=feature_transform).to(device=device)
    
    
else:
    print("dataset name not valid !!!! ")

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchsummary import summary

#vgg = models.vgg16()

prune_rate = 0
final_prune_rate = 99
bias = False
bestacc1 = []

#checkdir(f"{os.getcwd()}/saves/pointnet/modelnet40")
#weight_dir = f"{os.getcwd()}/saves/pointnet/modelnet40/init_for_lth.pth"

model0 = PointNetCls(k=num_classes, feature_transform=feature_transform).to(device=device)
#if os.path.exists(weight_dir):
#initial_state_dict = torch.load(weight_dir)
#model0.load_state_dict(initial_state_dict)
#else:
initial_state_dict = copy.deepcopy(model0.state_dict())
#checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/")
#torch.save(model0.state_dict(), weight_dir)
classifier.load_state_dict(initial_state_dict)
classifier.to(device)
best_model = classifier
# Optimizer and Scheduler
# Optimizer and Scheduler
optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
best_test_accuracy2 = 0.0  # Initialize the best accuracy variable
# Training loop

while (prune_rate < final_prune_rate):
    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    checkdir(f"{os.getcwd()}/saves/pointnet/{args.dataset}/full")
    # checkdir(f"{os.getcwd()}/saves/{args.arch_type}/{args.dataset}/{pt}/B")
    #torch.save(classifier.state_dict(),f"{os.getcwd()}/saves/pointnet/{args.dataset}full/{prune_rate:.2f}_model_{pt}.pth")
    #optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    if not pruning_round == 0:
        print(pruning_round)
        classifier = lth_local_unstructured_pruning(classifier,num_classes, bias=bias, conv=0.20, linear=0.20, output=0, batchnorm=0)#lth_local_unstructured_pruning(classifier,num_classes, bias=bias, conv=0.99, linear=0.99, output=0, batchnorm=0)#lth_global_unstructured_pruning(classifier, num_classes, bias=bias, conv=0.99, linear=0.99, output=0, batchnorm=0)lth_local_unstructured_pruning(classifier,num_classes, bias=bias, conv=0.10, linear=0.10, output=0, batchnorm=0)
        #classifier = lth_global_unstructured_pruning(classifier, num_classes, bias=bias, conv=0.99, linear=0.99, output=0.0 ,batchnorm=0)
        classifier = weight_rewinding(classifier, model0, bias=bias)
        prune_rate = prune_rate_calculator(classifier, bias=bias)
    
    best_test_accuracy = 0.0  # Initialize the best accuracy variable
    #best_epoch = 0
    n_epochs = 80
    best_epoch = 0
    from torch import nn
    for name, module in classifier.named_modules():
        if isinstance(module, nn.Linear):
            print(module.weight)
    print("\nModel's Parameters:")
    #for param in classifier.parameters():
        #print(param[0])
    print_nonzeros(classifier)
        
            
    for epoch in range(n_epochs):
        # Training
        classifier.train()
        #prune_rate = prune_rate_calculator(classifier, bias=bias)
        #print(prune_rate)
        total_train_loss = 0.0
        total_correct_train = 0
        total_time_train = 0.0
        num_batches_train = 0
        total_samples_train = 0
        '''
        for name, module in classifier.named_modules():
            if isinstance(module, nn.Linear):
                print(module.weight)
        '''

        EPS = 1e-6
        from weights import print_nonzeros
        progress_bar_train = tqdm(enumerate(dataloader, 0), total=len(dataloader), desc="Train Epoch   #%d" % (epoch), ncols=0)
        for i, data in progress_bar_train:
            start_time = time.time()
            #pr = print_nonzeros(classifier, bias=False)
            #print(pr)
            #points, target = data[0].to(device).float(), data[1].to(device)
            points, target = data[0].to(device).float(), data[1].squeeze().to(device)
            points = points.transpose(2, 1)

            optimizer.zero_grad()
            pred, trans, trans_feat = classifier(points)
            #target = target.squeeze()
            loss = F.nll_loss(pred, target)
	    
            if feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            
            loss.backward()
            
            for name, p in classifier.named_parameters():
                #print(name)
                if 'weight' in name:
                    #print(p.data)
                    tensor = p.data
                    grad_tensor = p.grad
                    grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                    p.grad.data = grad_tensor
                
            
            optimizer.step()
            #prune_rate = prune_rate_calculator(classifier, bias=bias)
            #print(prune_rate)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()

            end_time = time.time()

            total_train_loss += loss.item()
            total_correct_train += correct.item()
            total_time_train += (end_time - start_time)
            num_batches_train += 1
            total_samples_train += points.size(0)

        scheduler.step()
        avg_train_loss = total_train_loss / num_batches_train
        avg_train_accuracy = (total_correct_train / float(total_samples_train)) * 100
        avg_time_per_batch_train = total_time_train / num_batches_train
        train_result = '[Epoch %d] Avg train loss: %.3f | Avg accuracy: %.3f | Avg time/batch: %.3fs | Prune Rate: %.3f' % (epoch, avg_train_loss, avg_train_accuracy, avg_time_per_batch_train,prune_rate)
        print(colored(train_result, "blue"))

        # Testing
        classifier.eval()
        total_test_loss = 0.0
        total_correct_test = 0
        num_batches_test = 0
        total_samples_test = 0
        
        progress_bar_test = tqdm(enumerate(testdataloader, 0), total=len(testdataloader), desc="Test Epoch    #%d" % (epoch), ncols=0)
        with torch.no_grad():
            for i, data in progress_bar_test:
                #points, target = data[0].to(device).float(), data[1].to(device)
                points, target = data[0].to(device).float(), data[1].squeeze().to(device)
                points = points.transpose(2, 1)
		
                pred, _, _ = classifier(points)
                #target = target.squeeze()
                loss = F.nll_loss(pred, target)

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()

                total_test_loss += loss.item()
                total_correct_test += correct.item()
                num_batches_test += 1
                total_samples_test += points.size(0)

        avg_test_loss = total_test_loss / num_batches_test
        avg_test_accuracy = (total_correct_test / float(total_samples_test)) * 100
        if avg_test_accuracy > best_test_accuracy:
            best_test_accuracy = avg_test_accuracy
            best_epoch = epoch
            best_model = classifier
            #torch.save(classifier.state_dict(), '%s/best_cls_model.pth' % out_folder)  # Save best model

         # Print testing results
        test_result = '[Epoch %d] Avg test loss: %.3f | Avg accuracy: %.3f' % (epoch, avg_test_loss, avg_test_accuracy)
        
        print(colored(test_result, "green"))
    bestacc1.append(best_test_accuracy)
    if best_test_accuracy2<best_test_accuracy:
            best_test_accuracy2 = best_test_accuracy
    pruning_round += 1
    classifier = best_model
    print(colored('Best Epoch: {} with Test Accuracy: {:.3f} | Best Test Accuracy: {:.3f}'.format(best_epoch, best_test_accuracy,best_test_accuracy2), 'yellow'))

    #print(colored('[Epoch %d] Avg test loss: %.3f | Avg accuracy: %.3f' % (epoch, avg_test_loss, avg_test_accuracy), "green"))

    #torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (out_folder, epoch))
    
checkdir(f"{os.getcwd()}/dumps/lth/pointnet/{args.dataset}/local/full")
import numpy as np
np_bestacc1 = np.array([x for x in bestacc1])
np_bestacc1.dump(f"{os.getcwd()}/dumps/lth/pointnet/{args.dataset}/local/full/all_accuracy.dat")   
# all_loss2.dump(f"{os.getcwd()}/dumps/{pt}/{args.arch_type}/{args.dataset}/B/{pt}_all_loss_{comp2}.dat")
#print(colored('Best Epoch: {} with Test Accuracy: {:.3f}'.format(best_epoch, best_test_accuracy), 'yellow'))
# Print the best epoch and its accuracy
print(colored('Best Epoch: {} with Test Accuracy: {:.3f}'.format(best_epoch, best_test_accuracy), 'yellow'))



