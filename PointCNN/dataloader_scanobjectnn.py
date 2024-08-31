
import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_scanobjectnn_data(partition):
    # BASE_DIR= "G:\Scanobject Dataset_h5_format"
    # BASE_DIR = '/home/ece-desm//Ismail/ICPR_24/ScanObjectNN_Dataset'        # Rtx4090
    BASE_DIR= '/shafinSSD/Amrijit/ScanObjectNN_Dataset'                       # Rtx3090
    all_data = []
    all_label = []

    h5_name = os.path.join(BASE_DIR, 'h5_files', 'main_split', partition + '_objectdataset_augmentedrot_scale75.h5')
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNN(Dataset):
    def __init__(self, num_points, partition='training'):
        self.data, self.label = load_scanobjectnn_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(2500)
    test = ScanObjectNN(2500, 'test')
    # for data, label in train:                     # uncomment to see each shape size and labels
    #     print(data.shape)
    #     print(label)
# print("train size: ",train.__len__())
# print("test size: ",test.__len__())
