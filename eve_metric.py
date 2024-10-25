import torch

def calculate_mae(pred, true):
    """计算平均绝对误差(MAE)"""
    return torch.mean(torch.abs(pred - true)).item()

def calculate_mse(pred, true):
    """计算平均绝对误差(MAE)"""
    return torch.mean(torch.pow(true - pred, 2)).item()

def compute_bias(y_true, y_pred):
    bias = (y_pred - y_true).mean().item()
    return bias