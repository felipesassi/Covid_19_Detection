import torch
from utils.utils import show_training_progress

class Controller():
    def __init__(self, model, optimizer=None, loss=None, metric=None, train_data=None, validation_data=None, epochs=None, device=None, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer 
        self.loss = loss
        self.metric = metric
        self.train_data = train_data
        self.validation_data = validation_data
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.device = device

    def train(self):
        sig = nn.Sigmoid()
        for epoch in range(self.epochs):
            print("Epoch {}" .format(epoch))
            self.model.train()
            total_loss = 0
            self.metric.reset_value()
            for i, (x, y) in enumerate(self.train_data):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_out = self.model(x)
                loss_value = self.loss(y_out, y.float())
                loss_value.backward()
                self.optimizer.step()
                total_loss = total_loss + loss_value.item()
                y_out = sig(y_out)
                self.metric.add_value(y, y_out)
                self.metric.compute_metric()
                show_training_progress(self.metric.get_auc_metric(), i, len(self.train_data), True)
            if self.lr_scheduler != None:
                self.lr_scheduler.step(total_loss)
            print("")
            self.model.eval()
            total_loss = 0 
            self.metric.reset_value()
            with torch.no_grad():
                for i, (x, y) in enumerate(self.validation_data):
                    x, y = x.to(self.device), y.to(self.device)
                    y_out = self.model(x)
                    loss_value = self.loss(y_out, y.float())
                    total_loss = total_loss + loss_value.item()
                    y_out = sig(y_out)
                    self.metric.add_value(y, y_out)
                    self.metric.compute_metric()
                    show_training_progress(self.metric.get_auc_metric(), i, len(self.validation_data), False)
            print("")

        def save(self, train_mode=False):
            torch.save(self.model.state_dict(), "model.pth")

        def load(self, train_mode=False):
            self.model.load_state_dict(torch.load("model.pth"))

if __name__ == "__main__":
    pass