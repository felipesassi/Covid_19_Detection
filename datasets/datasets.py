import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2 as cv

class Data_Generator(Dataset):
    def __init__(self, data, labels, base_dir, transformer=None, train=True):
        self.data = data
        self.labels = labels
        self.base_dir = base_dir
        self.transformer = transformer
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        directory = self.base_dir + self.data[idx]
        img = Image.open(directory)
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.transformer(image = img)["image"]
        if self.train == True:
            label = self.labels[idx]
            return img, torch.tensor(label).view(-1)
        else:
            return img

def generate_train_validation_dataloader(data_train, data_train_labels, transformer_train, data_val, data_val_labels, transformer_val, batch_size, base_dir):
    train_loader = DataLoader(Data_Generator(data_train, data_train_labels, base_dir, transformer_train), batch_size)
    validation_loader = DataLoader(Data_Generator(data_val, data_val_labels, base_dir, transformer_val), batch_size)
    return train_loader, validation_loader

if __name__ == "__main__":
    pass