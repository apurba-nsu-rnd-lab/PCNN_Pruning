

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

# pass the params-

parser = argparse.ArgumentParser(description='Point Cloud Classification')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--dataset', type=str, default='modelnet40',
                        choices=['modelnet40', 'scanobjectnn', 'shapenetcore'])
parser.add_argument('--folder', type=str, default='m40_check',
                        choices=['m40', 'scan', 'shape'])
args = parser.parse_args()


# parameters for all datasets--
random_seed = 42
torch.manual_seed(random_seed)

# Hyperparameters
batch_size = 32           
n_epochs = 250
feature_transform = True
lr = 0.001
weight_decay = 1e-4

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############
if(args.dataset=='modelnet40'):
    dataset_name = 'modelnet40'
    #root = 'G:\ModelNet40 Dataset'  # <<<<<<< select root 
    # root = '/home/ece-desm/Ismail/ICPR_24/ModelNet40_Dataset'
    root= '/shafinSSD/Amrijit/ModelNet40_Dataset'
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
    #root = '/home/ece-desm//Ismail/ICPR_24/ShapeNetCore_Dataset'
    root= '/shafinSSD/Amrijit/ShapeNetCoreV2_ Dataset'
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


# Optimizer and Scheduler
optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

# Training loop
for epoch in range(n_epochs):
    # Training
    classifier.train()
    total_train_loss = 0.0
    total_correct_train = 0
    total_time_train = 0.0
    num_batches_train = 0
    total_samples_train = 0
    best_epoch = 0


    progress_bar_train = tqdm(enumerate(dataloader, 0), total=len(dataloader), desc="Train Epoch   #%d" % (epoch), ncols=0)
    for i, data in progress_bar_train:
        start_time = time.time()

        points, target = data[0].to(device).float(), data[1].to(device)
        points = points.transpose(2, 1)

        optimizer.zero_grad()
        pred, trans, trans_feat = classifier(points)
        target = target.squeeze()
        loss = F.nll_loss(pred, target)

        if feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        loss.backward()
        optimizer.step()

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
    train_result = '[Epoch %d] Avg train loss: %.3f | Avg accuracy: %.3f | Avg time/batch: %.3fs' % (epoch, avg_train_loss, avg_train_accuracy, avg_time_per_batch_train)
    print(colored(train_result, "blue"))

    # Testing
    classifier.eval()
    total_test_loss = 0.0
    total_correct_test = 0
    num_batches_test = 0
    total_samples_test = 0
    best_test_accuracy = 0
    progress_bar_test = tqdm(enumerate(testdataloader, 0), total=len(testdataloader), desc="Test Epoch    #%d" % (epoch), ncols=0)
    with torch.no_grad():
        for i, data in progress_bar_test:
            points, target = data[0].to(device).float(), data[1].to(device)
            points = points.transpose(2, 1)

            pred, _, _ = classifier(points)
            target = target.squeeze() 
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
        #torch.save(classifier.state_dict(), '%s/best_cls_model.pth' % out_folder)  # Save best model

     # Print testing results
    test_result = '[Epoch %d] Avg test loss: %.3f | Avg accuracy: %.3f' % (epoch, avg_test_loss, avg_test_accuracy)
    print(colored(test_result, "green"))

    #print(colored('[Epoch %d] Avg test loss: %.3f | Avg accuracy: %.3f' % (epoch, avg_test_loss, avg_test_accuracy), "green"))

    #torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (out_folder, epoch))

# Print the best epoch and its accuracy
print(colored('Best Epoch: {} with Test Accuracy: {:.3f}'.format(best_epoch, best_test_accuracy), 'yellow'))




