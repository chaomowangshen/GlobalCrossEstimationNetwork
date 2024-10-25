import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from time import time
from eve_metric import calculate_mae,calculate_mse,compute_bias
import os
import pandas as pd
import sys
#将上级目录作为根目录，这样就可以直接import data_loader等
sys.path.append('..')
from model.cen_qb import CEN_QB,EarlyStopping
from model.cen_qb_z import CEN_QB as CEN_QB_Z
from model.cen_qb_z import train_model_qb as train_model_qb_z
from model.cen_qb_z import train_one_epoch_qb as train_one_epoch_qb_z

def train_one_epoch_qb(model,res_mat,optimizer, device,loss_fn):
    res_mat = res_mat.clone().to(device)
    res_mat_item = res_mat.clone().T.to(device)
    label = res_mat.flatten().to(device)
    model.train()
    optimizer.zero_grad()
    pred = model(res_mat, res_mat_item).flatten()
    loss = loss_fn(pred, label)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_qb(z_true, a_true, b_true,z,a,b):
    mae_z = calculate_mae(z.flatten(), z_true.flatten())
    mae_a = calculate_mae(a.flatten(), a_true.flatten())
    mae_b = calculate_mae(b.flatten(), b_true.flatten())
    mse_z = calculate_mse(z.flatten(), z_true.flatten())
    mse_a = calculate_mse(a.flatten(), a_true.flatten())
    mse_b = calculate_mse(b.flatten(), b_true.flatten())
    bias_z = compute_bias(z_true.flatten(), z.flatten())
    bias_a = compute_bias(a_true.flatten(), a.flatten())
    bias_b = compute_bias(b_true.flatten(), b.flatten())
    return [[mae_z,mse_z,bias_z],[mae_a,mse_a,bias_a],[mae_b,mse_b,bias_b]]


def save_metrics_qb(start_time,end_time,times_list,metrics_list,z_true,a_true,b_true,z,a,b,n_person,n_item,rep,sample_size,strategy,sim='sim4'):
    elapsed_time = end_time - start_time
    times_list.append(elapsed_time)
    # Evaluate the model
    metric0 = evaluate_qb(z_true, a_true, b_true,z,a,b)
    metric0 = pd.DataFrame(metric0, columns=['MAE', 'MSE', 'Bias'], index=['z', 'a', 'b'])
    save_path = f'../metrics/{sim}/qb_batch/n_p_{n_person}/ni_{n_item}/rep{rep}/sample_size_{sample_size}/strategy_{strategy}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metric0.to_csv(f'{save_path}/metric{sample_size}_{strategy}.csv')
    metrics_list.append(metric0)
    return times_list,metrics_list


def train_model_qb(res_mat,n_item,n_person,depth,activation,sample_size,strategy,device, epochs, early_stopping_count,lr=0.001):
    if strategy == 'random':
        start_time = time()
        model1 = CEN_QB(n_item, sample_size, depth, activation).to(device)
        idx=torch.randperm(res_mat.shape[0])[:sample_size]
        res_mat0=res_mat[idx]
        res_mat_item0=res_mat0.T
        optimizer = optim.Adam(model1.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        early_stopping = EarlyStopping(patience=early_stopping_count, verbose=False)
        for epoch in range(epochs):
            loss = train_one_epoch_qb(model1,res_mat0,optimizer, device,loss_fn)
            if np.isnan(loss):
                print("Loss is NaN. Stopping training.")
                break
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {loss}')
            early_stopping(loss, model1)
            if early_stopping.early_stop:
                print(f'Early stopping on epoch {epoch}')
                break
        model1.load_state_dict(torch.load('best_model.pth'))
        a, b = model1.item_net(res_mat_item0.to(device))
        a = a.cpu()
        b = b.cpu()
        a_pd=pd.DataFrame(a.cpu().detach().numpy())
        a_pd.to_csv('a.csv',index=False, header=False)
        b_pd=pd.DataFrame(b.cpu().detach().numpy())
        b_pd.to_csv('b.csv',index=False, header=False)
        a_csv_path='a.csv'
        b_csv_path='b.csv'
        num_chunks = (res_mat.shape[0]+sample_size-1)//sample_size
        thetas=[]
        for i in range(num_chunks):
            start_idx=i*sample_size
            end_idx=min((i+1)*sample_size,res_mat.shape[0])
            res_mat_chunk=res_mat[start_idx:end_idx]
            model2 = CEN_QB_Z(a_csv_path, b_csv_path, n_item, depth, activation).to(device)
            optimizer = optim.Adam(model2.parameters(), lr=lr)
            loss_fn = nn.BCELoss()
            early_stopping = EarlyStopping(patience=early_stopping_count, verbose=False)
            for epoch in range(epochs):
                loss=train_one_epoch_qb_z(model2,res_mat_chunk,optimizer,device,loss_fn)
                if np.isnan(loss):
                    print("Loss is NaN. Stopping training.")
                    break
                '''if epoch % 100 == 0:
                    print(f'Epoch {epoch}/{epochs}, Loss: {loss}')'''
                early_stopping(loss, model2)
                if early_stopping.early_stop:
                    print(f'Early stopping on epoch {epoch}')
                    break
            model2.load_state_dict(torch.load('best_model.pth'))
            thetas.append(model2.person_net(res_mat_chunk.to(device)).cpu())
        thetas_flat = torch.cat([theta.flatten() for theta in thetas])
        os.remove('a.csv')
        os.remove('b.csv')
        end_time = time()
    else:
        num_ave=5
        start_time = time()
        ass=[]
        bss=[]
        for n in range(num_ave):
            model1 = CEN_QB(n_item, sample_size, depth, activation).to(device)
            idx = torch.randperm(res_mat.shape[0])[:sample_size]
            res_mat0 = res_mat[idx]
            res_mat_item0 = res_mat0.T
            optimizer = optim.Adam(model1.parameters(), lr=lr)
            loss_fn = nn.BCELoss()
            early_stopping = EarlyStopping(patience=early_stopping_count, verbose=False)
            for epoch in range(epochs):
                loss = train_one_epoch_qb(model1, res_mat0, optimizer, device, loss_fn)
                if np.isnan(loss):
                    print("Loss is NaN. Stopping training.")
                    break
                '''if epoch % 100 == 0:
                    print(f'Epoch {epoch}/{epochs}, Loss: {loss}')'''
                early_stopping(loss, model1)
                if early_stopping.early_stop:
                    print(f'Early stopping on epoch {epoch}')
                    break
            model1.load_state_dict(torch.load('best_model.pth'))
            a0, b0 = model1.item_net(res_mat_item0.to(device))
            a0 ,b0=a0.detach().cpu().numpy().flatten(),b0.detach().cpu().numpy().flatten()
            ass.append(a0)
            bss.append(b0)
        a=torch.Tensor(np.mean(ass,axis=0))
        b = torch.Tensor(np.mean(bss, axis=0))
        a_pd = pd.DataFrame(a.numpy())
        a_pd.to_csv('a.csv',index=False, header=False)
        b_pd = pd.DataFrame(b.numpy())
        b_pd.to_csv('b.csv',index=False, header=False)
        a_csv_path = 'a.csv'
        b_csv_path = 'b.csv'
        num_chunks = (res_mat.shape[0] + sample_size - 1) // sample_size
        thetas = []
        for i in range(num_chunks):
            start_idx = i * sample_size
            end_idx = min((i + 1) * sample_size, res_mat.shape[0])
            res_mat_chunk = res_mat[start_idx:end_idx]
            model2 = CEN_QB_Z(a_csv_path, b_csv_path, n_item, depth, activation).to(device)
            optimizer = optim.Adam(model2.parameters(), lr=lr)
            loss_fn = nn.BCELoss()
            early_stopping = EarlyStopping(patience=early_stopping_count, verbose=False)
            for epoch in range(epochs):
                loss = train_one_epoch_qb_z(model2, res_mat_chunk, optimizer, device, loss_fn)
                if np.isnan(loss):
                    print("Loss is NaN. Stopping training.")
                    break
                '''if epoch % 100 == 0:
                    print(f'Epoch {epoch}/{epochs}, Loss: {loss}')'''
                early_stopping(loss, model2)
                if early_stopping.early_stop:
                    print(f'Early stopping on epoch {epoch}')
                    break
            model2.load_state_dict(torch.load('best_model.pth'))
            thetas.append(model2.person_net(res_mat_chunk.to(device)).cpu())
        thetas_flat = torch.cat([theta.flatten() for theta in thetas])
        os.remove('a.csv')
        os.remove('b.csv')
        end_time = time()
    return a,b,thetas_flat,start_time,end_time





