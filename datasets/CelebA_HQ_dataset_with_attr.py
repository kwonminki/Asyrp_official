from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os
import torch
import natsort


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, test_nums=10000, train=True):
        img_dir = "/hdd1/datasets/CelebAMask-HQ/CelebA-HQ-img"
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)
        self.img_files = natsort.natsorted(self.img_files)
        if test_nums is not None:
            if train:
                self.img_files = self.img_files[:-test_nums]
            else:
                self.img_files = self.img_files[-test_nums:]
        self.transform = transform

        file_path = "/hdd1/datasets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
        self.attr_list = []
        self.attr = []
        self.file_name_list = []

        flag = False
    
        with open(file_path, "r") as f:
            line_num = 0
            for line in f:
                if line_num == 0:
                    line_num += 1
                    continue
                elif line_num == 1:
                    line_list = line.split(' ')
                    self.attr_list = line_list
                    line_num += 1
                else:
                    line_list = line[:-1].split(' ')
                    file_name = line_list[0]
                    if file_name == self.img_files[0]:
                        flag = True
                    if not flag:
                        continue
                    line_num += 1
                    self.file_name_list.append(file_name)
                    line_list = line_list[2:]
                    tmp_list = []
                    for value in line_list:
                        if value == '1':
                            tmp_list.append(1)
                        elif value == '-1':
                            tmp_list.append(0)
                        # attr_dict[key] = value
                    self.attr.append(tmp_list)
                
                if line_num == len(self.img_files) + 3:
                    break
        
        for ii, jj in zip(self.img_files, self.file_name_list):
            if ii != jj:
                print("Error")
        
                import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        attr = torch.Tensor(self.attr[idx])

        return image, attr

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        self.resolution = resolution
        self.transform = transform

        file_path = "/hdd1/datasets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
        self.attr_list = []
        self.attr = []
    
        with open(file_path, "r") as f:
            line_num = 0
            for line in f:
                if line_num == 0:
                    line_num += 1
                    continue
                elif line_num == 1:
                    line_list = line.split(' ')
                    self.attr_list = line_list
                    line_num += 1
                else:    
                    line_list = line.split(' ')[2:]
                    tmp_list = []
                    for key, value in zip(self.attr_list, line_list):
                        if value == '1':
                            tmp_list.append(1)
                        elif value == '-1':
                            tmp_list.append(0)
                        # attr_dict[key] = value
                    self.attr.append(tmp_list)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)
        attr = torch.Tensor(self.attr[index])

        return img, attr


################################################################################

def get_celeba_dataset_attr(data_root, config):
    train_transform = tfs.Compose([tfs.Resize((256, 256)), tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([tfs.Resize((256, 256)), tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    train_dataset = CustomImageDataset(os.path.join(data_root, 'LMDB_train'),
                                           train_transform, train=True)
    test_dataset = CustomImageDataset(os.path.join(data_root, 'LMDB_test'),
                                          test_transform, train=False)


    return train_dataset, test_dataset



