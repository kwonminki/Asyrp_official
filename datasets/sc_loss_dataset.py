from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os



class SemanticConsistencyDataset(Dataset):
    def __init__(self, path, transform= None, resolution=256):
        super().__init__()
        self.edit_dir = os.path.join(path,'edit_image')
        self.recon_dir = os.path.join(path,'recon_image')
        self.edit_list = os.listdir(self.edit_dir)
        self.transform = transform
        self.resolution = resolution

    def __getitem__(self, index):
        img_name = self.edit_list[index]
        print(img_name)
        edit_path = os.path.join(self.edit_dir, img_name)
        recon_path = os.path.join(self.recon_dir, img_name)

        edit_img = Image.open(edit_path)
        recon_img = Image.open(recon_path)

        edit_img = edit_img.resize((self.resolution, self.resolution))
        recon_img = recon_img.resize((self.resolution, self.resolution))

        if self.transform is not None:
            edit_img = self.transform(edit_img)
            recon_img = self.transform(recon_img)
        
        return edit_img, recon_img

    def __len__(self):
        return len(self.edit_list)