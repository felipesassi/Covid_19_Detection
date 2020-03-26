import torch
import torch.nn
import torch.nn.functional as F
import sys
import json
from sklearn.utils import shuffle

def get_device():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("{} is the device available." .format(device))
    return device

def clear_gpu_memory(device="cuda:0"):
    import gc
    gc.collect()
    memory_available = torch.cuda.memory_reserved(device)/1024**2
    print("Memory before cleaning: {} MB." .format(memory_available))
    torch.cuda.empty_cache()
    memory_available = torch.cuda.memory_reserved(device)/1024**2
    print("Memory after cleaning: {} MB." .format(memory_available))

def read_parameters():
  with open("configs/config.json", "r") as file:
    data = file.read()
  return json.loads(data)

def separate_train_val(data):
  data = shuffle(data, random_state = 0).reset_index(drop = True)
  n = data.shape[0]
  n1 = int(0.75*n)
  n2 = n - n1
  return data[:n1].reset_index(drop = True), data[n1:].reset_index(drop = True)

def show_training_progress(metric, index, batch_size, train):
    if train == False:
        txt_1 = "V - "
    else:
        txt_1 = "T - "
    txt_2 = "Metric: {:.2f}%" .format(metric)
    progress = int(100*(index + 1)/batch_size)
    progress_bar = progress*"#" + (100 - progress)*"-" + " - "
    txt = "\r" + txt_1 + progress_bar + txt_2
    sys.stdout.write(txt)

if __name__ == "__main__":
    pass