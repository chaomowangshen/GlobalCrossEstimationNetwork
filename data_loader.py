import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# 读取CSV文件并转换为PyTorch张量
def load_data(file_name, dtype=torch.float32):
    data = pd.read_csv(file_name, header=None, index_col=False)
    data = torch.tensor(data.values, dtype=dtype)
    return data


