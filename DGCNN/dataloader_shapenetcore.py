
import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud

class Dataset_shapenet(data.Dataset):
    def __init__(self, root, dataset_name='shapenetcorev2', class_choice=None,
            num_points=2048, split='train', load_name=True, load_file=True,
            random_rotate=False, random_jitter=False, random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 
            'modelnet10', 'modelnet40', 'shapenetpartpart']
        assert num_points <= 2048        

        if dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        self.root = os.path.join(root, dataset_name + '_hdf5_2048')  # <<<<<<<<<<<   watch path here [ root + name + _hdf5_2048 ]
        self.dataset_name = dataset_name
        self.class_choice = class_choice
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        
        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train', 'trainval', 'all']:   
            self.get_path('train')
        if self.dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            if self.split in ['val', 'trainval', 'all']: 
                self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        data, label = self.load_h5py(self.path_h5py_all)

        if self.load_name or self.class_choice != None:
            self.name = np.array(self.load_json(self.path_name_all))  # load label name

        if self.load_file:
            self.file = np.array(self.load_json(self.path_file_all))  # load file name
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 

        if self.class_choice != None:
            indices = (self.name == class_choice)
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.name = self.name[indices]
            if self.load_file:
                self.file = self.file[indices]

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '%s*.h5' % type)
        paths = glob(path_h5py)
        paths_sort = [os.path.join(self.root, type + str(i) + '.h5') for i in range(len(paths))]
        self.path_h5py_all += paths_sort
        if self.load_name:
            paths_json = [os.path.join(self.root, type + str(i) + '_id2name.json') for i in range(len(paths))]
            self.path_name_all += paths_json
        if self.load_file:
            paths_json = [os.path.join(self.root, type + str(i) + '_id2file.json') for i in range(len(paths))]
            self.path_file_all += paths_json
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name
        if self.load_file:
            file = self.file[item]  # get file name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)

        # return point_set, label, name, file
        return point_set, label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    # root = 'G:\ShapeNetCoreV2'  # <<<<<<< select root
    root = '/home/ece-desm//Ismail/ICPR_24/ShapeNetCore_Dataset'

    # choose dataset name from 'shapenetcorev2', 'shapenetpart', 'modelnet40' and 'modelnet10'
    dataset_name = 'shapenetcorev2'

    # choose split type from 'train', 'test', 'all', 'trainval' and 'val'
    # only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'

    train = Dataset_shapenet(root=root, dataset_name=dataset_name, num_points=2048, split='trainval')
    test = Dataset_shapenet(root=root, dataset_name=dataset_name, num_points=2048, split='test')
    print("datasize:", train.__len__())
    print("datasize:", test.__len__())
    # points, label,name,file= train[70]
    # print(name)
    # print(label)
    # print("all ok ")
    
# if __name__ == '__main__':
#     train()
