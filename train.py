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
from metrics.metrics import Accuracy_Metric, ROC_AUC_Score
import albumentations as A
from albumentations import pytorch

if __name__ == "__main__":
    df = pd.read_csv("data/metadata.csv")
    df_train, df_val = separate_train_val(df)
    configs = read_parameters()
    device = get_device()
    transform = A.Compose([
                       A.Resize(512, 512, p = 1),
                       A.OneOf([
                                A.RGBShift(p = 1),
                                A.Blur(p = 1),
                                A.RandomGamma(p = 1),
                                A.RandomBrightness(p = 1),
                                A.RandomContrast(p = 1),
                                ]),
                        A.OneOf([
                                 A.VerticalFlip(p = 1),
                                 A.HorizontalFlip(p = 1)
                        ]),
                       A.CoarseDropout(max_holes = 24),
                       A.Normalize(p = 1),
                       pytorch.ToTensorV2()
    ])
    transform_val_test = A.Compose([
                                A.Resize(height = 512, width = 512, p = 1),
                                A.Normalize(p = 1),
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
                                                                        "data/data/train/")
        EFNet = EF_Net().to(device)
        Loss = nn.BCEWithLogitsLoss()
        Optimizer = optim.Adam(EFNet.parameters(),
                            lr = configs["train_parameters"]["learning_rate"])
        Metrics = ROC_AUC_ScoreC()
        Control = Controller(model = EFNet,
                    optimizer = Optimizer,
                    loss = Loss,
                    metric = Metrics,
                    train_data = train_gen,
                    validation_data = val_gen,
                    epochs = configs["train_parameters"]["epochs"],
                    device = device,
                    lr_scheduler = None)
        Control.train()