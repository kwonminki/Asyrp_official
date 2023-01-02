from .AFHQ_dataset import get_afhq_dataset
from .CelebA_HQ_dataset import get_celeba_dataset
from .LSUN_dataset import get_lsun_dataset
from torch.utils.data import DataLoader
from .IMAGENET_dataset import get_imagenet_dataset
from .CelebA_HQ_dataset_dialog import get_celeba_dialog_dataset
from .CelebA_HQ_dataset_with_attr import get_celeba_dataset_attr

from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, test_nums=None, train=True):
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)
        if test_nums is not None:
            if train:
                self.img_files = self.img_files[:-test_nums]
            else:
                self.img_files = self.img_files[-test_nums:]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image

def get_dataset(dataset_type, dataset_paths, config, target_class_num=None, gender=None):
    # if category is CUSTOM, get images from custom arg path
    if config.data.category == "CUSTOM":
        train_dataset = CustomImageDataset(dataset_paths['custom_train'], transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        test_dataset = CustomImageDataset(dataset_paths['custom_test'], transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return train_dataset, test_dataset

    if dataset_type == 'AFHQ':
        train_dataset, test_dataset = get_afhq_dataset(dataset_paths['AFHQ'], config)
    elif dataset_type == "LSUN":
        train_dataset, test_dataset = get_lsun_dataset(dataset_paths['LSUN'], config)
    elif dataset_type == "CelebA_HQ-attr":
        train_dataset, test_dataset = get_celeba_dataset_attr(dataset_paths['CelebA_HQ'], config)
    elif dataset_type == "CelebA_HQ":
        train_dataset, test_dataset = get_celeba_dataset(dataset_paths['CelebA_HQ'], config)
    elif dataset_type == "CelebA_HQ_Dialog":
        train_dataset, test_dataset = get_celeba_dialog_dataset(dataset_paths['CelebA_HQ_Dialog'], config)
    elif dataset_type == "IMAGENET":
        train_dataset, test_dataset = get_imagenet_dataset(dataset_paths['IMAGENET'], config, class_num=target_class_num)
    elif dataset_type == "MetFACE":
        train_dataset = CustomImageDataset(os.path.join(dataset_paths['MetFACE'],'images'), transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), test_nums=500)
        test_dataset = CustomImageDataset(os.path.join(dataset_paths['MetFACE'],'images'), transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), test_nums=500, train=False)
    elif dataset_type == "FFHQ":
        train_dataset = CustomImageDataset(dataset_paths['FFHQ'], transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), test_nums=500)
        test_dataset = CustomImageDataset(dataset_paths['FFHQ'], transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), test_nums=500, train=False)
    else:
        raise ValueError

    return train_dataset, test_dataset


def get_dataloader(train_dataset, test_dataset, bs_train=1, num_workers=0, shuffle=False):
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs_train,
        drop_last=True,
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=True,
        sampler=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {'train': train_loader, 'test': test_loader}


