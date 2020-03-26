import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as numpy
from sklearn.metrics import roc_auc_score

class ROC_AUC_Score():
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.roc_value = 0

    def compute_metric(self):
        try:
            roc_value = roc_auc_score(self.y_true.cpu().detach().numpy(), self.y_pred.cpu().detach().numpy())
        except:
            roc_value = 0
        self.roc_value = roc_value

    def reset_value(self):
        self.y_true = None
        self.y_pred = None
        self.roc_value = 0

    def add_value(self, y_true, y_pred):
        if self.y_true == None:
            self.y_true = y_true
        else:
            self.y_true = torch.cat([self.y_true, y_true], dim = 0)
        if self.y_pred == None:
            self.y_pred = y_pred
        else:
            self.y_pred = torch.cat([self.y_pred, y_pred], dim = 0)

    def get_auc_metric(self):
        return self.roc_value

class Accuracy_Metric():
    def __init__(self):
        self.accuracy = 0
        self.accuracy_temp = 0

    def compute_metric(self, y_pred, y_true, batch_size, i):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred <= 0] = 0
        correct = (y_pred == y_true).sum()
        accuracy = 100*correct/batch_size
        self.accuracy_temp = self.accuracy_temp + accuracy
        self.accuracy = self.accuracy_temp/(i + 1)

    def reset_accuracy_value(self):
        self.accuracy = 0
        self.accuracy_temp = 0

    def get_accuracy_value(self):
        return self.accuracy

if __name__ == "__main__":
    pass