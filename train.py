import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from models.model import EF_Net
from models.controller import Controller
from utils.utils import get_device, read_parameters, separate_train_val
from metrics.metrics import Accuracy_Metric
import albumentations as A
from albumentations import pytorch

if __name__ == "__main__":
    df = pd.read_csv("data/metadata.csv")
    configs = read_parameters()
    device = get_device()
    transform = A.Compose([
                        A.Resize(256, 256, p = 1),
                        A.OneOf([
                                    A.Blur(p = 1),
                                    A.RandomGamma(p = 1),
                                    A.RandomBrightness(p = 1),
                                    A.RandomContrast(p = 1),
                                    ]),
                            A.OneOf([
                                    A.VerticalFlip(p = 1),
                            ]),
                        A.CoarseDropout(p = 0.5),
                        A.Normalize(p = 1),
                        pytorch.ToTensorV2()
    ])
    transform_val_test = A.Compose([
        A.Resize(height = 256, width = 256, p = 1.0),
        A.Normalize(p = 1.0),
        pytorch.ToTensorV2(),
    ])
    KF = StratifiedKFold()
    for train, val in KF.split(df["labels"].values, df["classes"].values):
        x_train, x_val = np.array(df["labels"].values)[train], np.array(df["labels"].values)[val]
        y_train, y_val = np.array(df["classes"].values)[train], np.array(df["classes"].values)[val],
        train_loader, val_loader = generate_train_validation_dataloader(x_train, 
                                                                        y_train,
                                                                        transform,
                                                                        x_val,
                                                                        y_val,
                                                                        transform_val_test
                                                                        configs["train_parameters"]["batch_size"], 
                                                                        "data/images/")
        EFNet = EF_Net().to(device)
        Loss = nn.CrossEntropyLoss()
        Optimizer = optim.Adam(EFNet.parameters(),
                            lr = configs["train_parameters"]["learning_rate"])
        Stepper = optim.lr_scheduler.ReduceLROnPlateau(Optimizer, patience = configs["train_parameters"]["patience"])
        Metrics = Accuracy_Metric()
        Control = Controller(model = EFNet,
                    optimizer = Optimizer,
                    loss = Loss,
                    metric = Metrics,
                    train_data = train_gen,
                    validation_data = val_gen,
                    epochs = configs["train_parameters"]["epochs"],
                    device = device,
                    lr_scheduler = Stepper)
        Control.train()