from torchvision import transforms
from skimage.color import rgb2lab, rgb2gray
from torch.utils.data import Dataset

import copy
import torch
import numpy as np
from PIL import Image
import os

class BlackWhite2Color(Dataset):
    def __init__(self, root, transform, mode='train'):
        self.root = root
        self.transform = transform
        self.mode = mode
        
        data_dir = os.path.join(root, mode)
        self.file_list = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.mode, self.file_list[idx])
        img = Image.open(img_path)
        
        if self.transform is not None:
            img_original = self.transform(img)

        # resize image to [56, 56, 3]
        img_resize = transforms.Resize(56)(img_original)
        img_lab = rgb2lab(img_resize)
        # get ab channel 
        img_ab = img_lab[:, :, 1:3]
        # [ab, H/4, W/4]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))  

        img_original = np.asarray(img_original)
        # get L channel, subtract 50 for mean-centering
        img_original = rgb2lab(img_original)[:,:,0] - 50.  
        # [224, 224, 1]
        img_original = torch.from_numpy(img_original)  
        
        return img_original, img_ab


def data_loader(root, batch_size=2, shuffle=True, img_size=224, mode='train'):    
    transform = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip() if mode =='train' else None,
                                   ])
    
    dset = BlackWhite2Color(root, transform, mode=mode)
    
    if batch_size == 'all':
        batch_size = len(dset)
        
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=0,
                                          drop_last=True)
    dlen = len(dset)
    
    return dloader, dlen


class ValidateDataset(Dataset):
    def __init__(self, root, transform, mode='test'):
        self.root = root
        self.transform = transform
        self.mode = mode
        
        data_dir = os.path.join(root, mode)
        self.file_list = os.listdir(data_dir)
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.mode, self.file_list[idx])
        image = Image.open(image_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        image = np.asarray(image)
        img_original = copy.deepcopy(image)
        image_gray = rgb2lab(image)[:, :, 0][:, :, None]
        # image input
        img_input = image_gray - 50.
        # [224, 224, 1]
        img_input = torch.from_numpy(img_input.transpose((2, 0, 1)))  
        
        return img_original, image_gray, img_input


def validate_loader(root, batch_size=2, shuffle=True, img_size=224, mode='Test'):    
    transform = transforms.Compose([
                                    transforms.Resize((img_size,img_size)),
                                   ])
    
    dset = ValidateDataset(root, transform, mode=mode)
    
    if batch_size == 'all':
        batch_size = len(dset)
        
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=0,
                                          drop_last=True)
    dlen = len(dset)
    
    return dloader, dlen