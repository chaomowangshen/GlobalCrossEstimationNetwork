import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from time import time
import math
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import warnings
# 忽视来自torch.nn.functional的特定警告
warnings.filterwarnings("ignore")
import sys
#将上级目录作为根目录，这样就可以直接import data_loader等
sys.path.append('..')
import os
#可以忽视下面划线，可以正常导入
from data_loader import load_data
from model.batch_cen_qb import train_model_qb as train_model_qb_batch
from model.batch_cen_qb import save_metrics_qb as save_metrics_qb_batch
from model.cen_qb_new import train_model_qb,save_metrics_qb
import argparse
def parse_activations(input_str):
    activations_dict = {}
    items = input_str.split(',')
    for item in items:
        try:
            key, value = item.split(':')
            activations_dict[int(key)] = value.strip()
        except ValueError:
            print(f"错误：无法解析 '{item}'. 正确的格式应该是 'key:value'。")
            exit(1)
    return activations_dict

def parse_depths(input_str):
    depths_dict = {}
    items = input_str.split(',')
    for item in items:
        try:
            key, value = item.split(':')
            depths_dict[int(key)] = int(value.strip())
        except ValueError:
            print(f"错误：无法解析 '{item}'. 正确的格式应该是 'key:value'。")
            exit(1)
    return depths_dict

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulations1")
    parser.add_argument('--n_persons_sim', nargs='+', type=int, default=[5000])
    parser.add_argument('--n_items_sim', nargs='+', type=int, default=[10,20,40,60])
    parser.add_argument('--rep_sim', type=int, default=100)
    parser.add_argument('--depths', type=str, default="10:1,20:3,40:3,60:3")
    parser.add_argument('--activations', type=str, default="10:sigmoid,20:tanh,40:tanh,60:tanh")
    parser.add_argument('--sample_sizes', nargs='+',type=int, default=[500,1000,1500])
    parser.add_argument('--strategies', nargs='+', type=str, default=['random', 'mean'])
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--early_stopping_count', type=int, default=30)
    args = parser.parse_args()
    return args


def run(n_person_sim,n_item_sim,n_reps,depths,activations,sample_sizes,strategies,batch_size,lr,epochs,early_stopping_count,sim='sim4'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for n_person in n_person_sim:#10000
        for n_item in n_item_sim:#10-60
            total_time=[]
            time_names=[]
            n_depth=depths[n_item]#选择该题目的深度
            activation=activations[n_item]#选择该题目的激活函数
            for sample_size in sample_sizes:
                for strategy in strategies:
                    times_list=[]
                    metrics_list=[]
                    timename=f'san_{sample_size}_str_{strategy}_item_{n_item}'
                    time_names.append(timename)
                    for rep in range(n_reps):
                        print(f"Running simulation4 {rep + 1}/{n_reps} with {n_person} persons and {n_item} items_with {sample_size} sample_size and {strategy} strategy")
                        # Load the data
                        res_mat = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/res_mat.csv')
                        z_true = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/z.csv')
                        a_true = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/a.csv')
                        b_true = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/b.csv')
                        # Initialize the model
                        a,b,z,start_time,end_time=train_model_qb_batch(res_mat,n_item,n_person,n_depth,activation,sample_size,strategy,device, epochs, early_stopping_count,lr)
                        times_list,metrics_list=save_metrics_qb_batch(start_time,end_time,times_list,metrics_list,z_true,a_true,b_true,z,a,b,n_person,n_item,rep,sample_size,strategy,sim=sim)
                
                    # Save the results
                    timese=np.array(times_list).mean()
                    total_time.append(timese)
                    #print(metrics_list)
                    metrics_mean=np.array(metrics_list).mean(axis=0)
                    metrics_mean=pd.DataFrame(metrics_mean,columns=['MAE','MSE','Bias'],index=['z','a','b'])
                    metrics_std=np.array(metrics_list).std(axis=0)
                    metrics_std=pd.DataFrame(metrics_std,columns=['MAE','MSE','Bias'],index=['z','a','b'])
                    save_path_mean=f'../metrics_summary/{sim}/qb_batch/n_p_{n_person}_ni_{n_item}'
                    save_path_std=f'../metrics_summary/{sim}/qb_batch/n_p_{n_person}_ni_{n_item}'
                    if not os.path.exists(save_path_mean):
                        os.makedirs(save_path_mean)
                    if not os.path.exists(save_path_std):
                        os.makedirs(save_path_std)
                    save_path_mean =save_path_mean+f'/sample_size_{sample_size}_strategy_{strategy}_MEAN.csv'
                    save_path_std=save_path_std+f'/sample_size_{sample_size}_strategy_{strategy}_STD.csv'
                    metrics_mean.to_csv(save_path_mean)
                    metrics_std.to_csv(save_path_std)
            times_list=[]
            metrics_list=[]
            timename=f'qb_item_{n_item}'
            time_names.append(timename)
            n_depth=3
            activation='tanh'
            for rep in range(n_reps):
                print(f"Running simulation4 {rep + 1}/{n_reps} with {n_person} persons and {n_item} items_with {n_depth} depth and {activation} activation")
                # Load the data
                res_mat = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/res_mat.csv')
                z_true = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/z.csv')
                a_true = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/a.csv')
                b_true = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/b.csv')
                model,start_time,end_time=train_model_qb(res_mat,n_item,n_person,n_depth,activation,device,epochs, early_stopping_count,batch_size,lr)
                times_list, metrics_list = save_metrics_qb(start_time, end_time, times_list, metrics_list, model,
                                                           res_mat, device, z_true, a_true, b_true, n_depth, activation,
                                                           n_person, n_item, rep,sim=sim)
            timese = np.array(times_list).mean()
            total_time.append(timese)
            metrics_mean = np.array(metrics_list).mean(axis=0)
            metrics_mean = pd.DataFrame(metrics_mean, columns=['MAE', 'MSE', 'Bias'], index=['z', 'a', 'b'])
            metrics_std = np.array(metrics_list).std(axis=0)
            metrics_std = pd.DataFrame(metrics_std, columns=['MAE', 'MSE', 'Bias'], index=['z', 'a', 'b'])
            save_path_mean = f'../metrics_summary/{sim}/qb/n_p_{n_person}_ni_{n_item}'
            save_path_std = f'../metrics_summary/{sim}/qb/n_p_{n_person}_ni_{n_item}'
            if not os.path.exists(save_path_mean):
                os.makedirs(save_path_mean)
            if not os.path.exists(save_path_std):
                os.makedirs(save_path_std)
            save_path_mean = save_path_mean + f'/qb_mean.csv'
            save_path_std = save_path_std + f'/qb_std.csv'
            metrics_mean.to_csv(save_path_mean)
            metrics_std.to_csv(save_path_std)
            total_time=pd.DataFrame(total_time,columns=['time'],index=time_names)
            save_path_time=f'../metrics_summary/{sim}/time'
            if not os.path.exists(save_path_time):
                os.makedirs(save_path_time)
            save_path_time=save_path_time+f'/n_p_{n_person}_n_i_{n_item}.csv'
            total_time.to_csv(save_path_time)

if __name__ == '__main__':
    args=parse_arguments()
    n_person_sim=args.n_persons_sim
    n_item_sim=args.n_items_sim
    n_reps=args.rep_sim
    depths=parse_depths(args.depths)
    activations=parse_activations(args.activations)
    batch_size=args.batch_size
    lr=args.lr
    epochs=args.epochs
    early_stopping_count=args.early_stopping_count
    run(n_person_sim,n_item_sim,n_reps,depths,activations,args.sample_sizes,args.strategies,batch_size,lr,epochs,early_stopping_count,sim='sim4')






















