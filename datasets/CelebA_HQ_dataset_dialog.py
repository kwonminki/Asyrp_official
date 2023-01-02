from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os
import pandas as pd

class MultiResolutionDialogDataset(Dataset):
    def __init__(self, path, transform, train=True, guid="Smiling"):
        guid = "Young"
        print("Image:", guid)
        self.path = path
        self.transform = transform
        if train:
            attrs = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "train_attr_list.txt"), sep=" ", header=None)
        else:
            attrs = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "val_attr_list.txt"), sep=" ", header=None)
        attrs = attrs.dropna(axis=1, how="all")
        attrs.columns = ["file_name", "Bangs", "Eyeglasses", "No_Beard", "Smiling", "Young"]

        condition_0 = (attrs[guid] == 0)
        condition_5 = (attrs[guid] == 5)

        attrs_0 = attrs[condition_0]
        attrs_5 = attrs[condition_5]

        self.min_num = min(len(attrs_0), len(attrs_5))
        
        self.attrs_0 = attrs_0.reset_index().loc[:self.min_num-1]
        self.attrs_5 = attrs_5.reset_index().loc[:self.min_num-1]


    def __len__(self):
        return self.min_num

    def __getitem__(self, index):

        filenum_0 = self.attrs_0.iloc[index]["file_name"]
        filenum_5 = self.attrs_5.iloc[index]["file_name"]

        filenum_0 = os.path.join(self.path, filenum_0)
        filenum_5 = os.path.join(self.path, filenum_5)
        
        img0 = Image.open(filenum_0)
        img0 = self.transform(img0)
        img5 = Image.open(filenum_5)
        img5 = self.transform(img5)

        return (img0, img5) # img0, img5


################################################################################

def get_celeba_dialog_dataset(data_root, config):
    train_transform = tfs.Compose([ tfs.Resize((config.data.image_size,config.data.image_size)),
                                    tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([ tfs.Resize((config.data.image_size,config.data.image_size)),
                                    tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    train_dataset = MultiResolutionDialogDataset(data_root,
                                           train_transform, train=True, guid="Smiling")
    test_dataset = None # MultiResolutionDialogDataset(os.path.join(data_root, 'LMDB_test'),
                         #                 test_transform, config.data.image_size, train=False, guid="Smiling")


    return train_dataset, test_dataset



