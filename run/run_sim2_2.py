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
from model.cen_init_z import CEN,train_one_epoch_init,evaluate_init,save_metrics,train_model_init
from model.cen_qb_z import CEN_QB,train_one_epoch_qb,evaluate_qb,save_metrics_qb,train_model_qb
import argparse
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulations1")
    parser.add_argument('--n_persons_sim', nargs='+', type=int, default=[100,500,1000])
    parser.add_argument('--n_items_sim', nargs='+', type=int, default=[10,20,40,60])
    parser.add_argument('--rep_sim', type=int, default=100)
    parser.add_argument('--depths', nargs='+', type=int, default=[1,3])
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--early_stopping_count', type=int, default=30)
    args = parser.parse_args()
    return args

def run(n_person_sim,n_item_sim,n_reps,depths,batch_size,lr,epochs,early_stopping_count,sim='sim2',res='res2_2'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)
    for n_person in n_person_sim:
        for n_item in n_item_sim:
            total_time=[]
            for n_depth in depths:
                activation='sigmoid'
                times_list=[]
                metrics_list=[]
                for rep in range(n_reps):
                    print(f"Running simulation2_2 init {rep + 1}/{n_reps} with {n_person} persons and {n_item} items_with {n_depth} depth and {activation} activation")
                    # Load the data
                    res_mat = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/res_mat.csv')
                    a_csv_path = f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/a.csv'
                    b_csv_path = f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/b.csv'
                    z_true=load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/z.csv')
                    # Initialize the model
                    model,start_time,end_time=train_model_init(res_mat,a_csv_path,b_csv_path,n_item,n_person,n_depth,activation,device, epochs, early_stopping_count,batch_size,lr)
                    times_list,metrics_list=save_metrics(start_time,end_time,times_list,metrics_list,model,res_mat,device,z_true,n_depth,activation,n_person,n_item,rep,sim,res)
                    # Save the results
                timese=np.array(times_list).mean()
                total_time.append(timese)
                metrics_mean=np.array(metrics_list).mean(axis=0)
                metrics_mean=pd.DataFrame(metrics_mean,columns=['MAE','MSE','Bias'],index=['z'])
                metrics_std=np.array(metrics_list).std(axis=0)
                metrics_std=pd.DataFrame(metrics_std,columns=['MAE','MSE','Bias'],index=['z'])
                save_path_mean=f'../metrics_summary/{sim}/{res}/init/n_p_{n_person}_ni_{n_item}'
                save_path_std=f'../metrics_summary/{sim}/{res}/init/n_p_{n_person}_ni_{n_item}'
                if not os.path.exists(save_path_mean):
                    os.makedirs(save_path_mean)
                if not os.path.exists(save_path_std):
                    os.makedirs(save_path_std)
                save_path_mean =save_path_mean+f'/depth_{n_depth}_act_{activation}_mean.csv'
                save_path_std=save_path_std+f'/depth_{n_depth}_act_{activation}_std.csv'
                metrics_mean.to_csv(save_path_mean)
                metrics_std.to_csv(save_path_std)

            for n_depth in depths:
                activation='tanh'
                times_list=[]
                metrics_list=[]
                for rep in range(n_reps):
                    print(f"Running simulation qb {rep + 1}/{n_reps} with {n_person} persons and {n_item} items_with {n_depth} hidden_dim and {activation} activation")
                    # Load the data
                    res_mat = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/res_mat.csv')
                    a_csv_path = f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/a.csv'
                    b_csv_path = f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/b.csv'
                    z_true = load_data(f'../data/{sim}/n_p_{n_person}/n_i_{n_item}/rep_{rep}/z.csv')
                    # Initialize the model
                    model,start_time,end_time=train_model_qb(res_mat,a_csv_path,b_csv_path,n_item,n_depth,activation,device, epochs, early_stopping_count,lr)
                    times_list,metrics_list=save_metrics_qb(start_time,end_time,times_list,metrics_list,model,res_mat,device,z_true,n_depth,activation,n_person,n_item,rep,sim,res)
                    # Save the results
                timese=np.array(times_list).mean()
                total_time.append(timese)
                metrics_mean=np.array(metrics_list).mean(axis=0)
                metrics_mean=pd.DataFrame(metrics_mean,columns=['MAE','MSE','Bias'],index=['z'])
                metrics_std=np.array(metrics_list).std(axis=0)
                metrics_std=pd.DataFrame(metrics_std,columns=['MAE','MSE','Bias'],index=['z'])
                save_path_mean=f'../metrics_summary/{sim}/{res}/qb/n_p_{n_person}_ni_{n_item}'
                save_path_std=f'../metrics_summary/{sim}/{res}/qb/n_p_{n_person}_ni_{n_item}'
                if not os.path.exists(save_path_mean):
                    os.makedirs(save_path_mean)
                if not os.path.exists(save_path_std):
                    os.makedirs(save_path_std)
                save_path_mean =save_path_mean+f'/depth_{n_depth}_act_{activation}_mean.csv'
                save_path_std=save_path_std+f'/depth_{n_depth}_act_{activation}_std.csv'
                metrics_mean.to_csv(save_path_mean)
                metrics_std.to_csv(save_path_std)
            total_time=pd.DataFrame(total_time,columns=['time'])
            save_path_time=f'../metrics_summary/{sim}/{res}/time'
            if not os.path.exists(save_path_time):
                os.makedirs(save_path_time)
            save_path_time=save_path_time+f'/n_p_{n_person}_n_i_{n_item}.csv'
            total_time.to_csv(save_path_time)

if __name__ == '__main__':
    args=parse_arguments()
    n_person_sim=args.n_persons_sim
    n_item_sim=args.n_items_sim
    n_reps=args.rep_sim
    depths=args.depths
    batch_size=args.batch_size
    lr=args.lr
    epochs=args.epochs
    early_stopping_count=args.early_stopping_count
    run(n_person_sim,n_item_sim,n_reps,depths,batch_size,lr,epochs,early_stopping_count)





















