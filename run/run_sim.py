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
from model.cen_init import CEN,train_one_epoch_init,evaluate_init,save_metrics,train_model_init
from model.cen_qb import CEN_QB,train_one_epoch_qb,evaluate_qb,save_metrics_qb,train_model_qb
from eve_metric import calculate_mae,calculate_mse,compute_bias
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulations1")
    parser.add_argument('--n_persons_sim', nargs='+', type=int, default=[100,500,1000])
    parser.add_argument('--n_items_sim', nargs='+', type=int, default=[30,60,90])
    parser.add_argument('--rep_sim', type=int, default=100)
    parser.add_argument('--depths', nargs='+', type=int, default=[1,3])
    parser.add_argument('--activations', nargs='+', type=str, default=['sigmoid','leakyrelu','tanh'])
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--early_stopping_count', type=int, default=30)
    args = parser.parse_args()
    return args

class EarlyStopping:
    def __init__(self, patience=30, verbose=False,delta=0.001):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss, model,filename='best_model.pth'):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model,filename)
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                if self.verbose:
                    print("Early stopping")
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model,filename)
            self.epochs_no_improve = 0

    def save_checkpoint(self, model,filename):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased. Saving model')
        torch.save(model.state_dict(),filename)





def run(n_person_sim1,n_item_sim1,n_reps,depths,activations,batch_size,lr,epochs,early_stopping_count):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for n_person in n_person_sim1:
        for n_item in n_item_sim1:
            total_time=[]
            for n_depth in depths:
                for activation in activations:
                    times_list=[]
                    metrics_list=[]
                    for rep in range(n_reps):
                        print(f"Running simulation init {rep + 1}/{n_reps} with {n_person} persons and {n_item} items_with {n_depth} depth and {activation} activation")
                        # Load the data
                        res_mat = load_data(f'../data/n_p_{n_person}/n_i_{n_item}/rep_{rep}/res_mat.csv')
                        z_true = load_data(f'../data/n_p_{n_person}/n_i_{n_item}/rep_{rep}/z.csv')
                        a_true = load_data(f'../data/n_p_{n_person}/n_i_{n_item}/rep_{rep}/a.csv')
                        b_true = load_data(f'../data/n_p_{n_person}/n_i_{n_item}/rep_{rep}/b.csv')
                        # Initialize the model
                        model,start_time,end_time=train_model_init(res_mat,n_item,n_person,n_depth,activation,device, epochs, early_stopping_count,batch_size,lr)
                        times_list,metrics_list=save_metrics(start_time,end_time,times_list,metrics_list,model,res_mat,device,z_true,a_true,b_true,n_depth,activation,n_person,n_item,rep)
                    # Save the results
                    timese=np.array(times_list).mean()
                    total_time.append(timese)
                    metrics_mean=np.array(metrics_list).mean(axis=0)
                    metrics_mean=pd.DataFrame(metrics_mean,columns=['MAE','MSE','Bias'],index=['z','a','b'])
                    metrics_std=np.array(metrics_list).std(axis=0)
                    metrics_std=pd.DataFrame(metrics_std,columns=['MAE','MSE','Bias'],index=['z','a','b'])
                    save_path_mean=f'../metrics_summary/sim1/init/n_p_{n_person}_ni_{n_item}'
                    save_path_std=f'../metrics_summary/sim1/init/n_p_{n_person}_ni_{n_item}'
                    if not os.path.exists(save_path_mean):
                        os.makedirs(save_path_mean)
                    if not os.path.exists(save_path_std):
                        os.makedirs(save_path_std)
                    save_path_mean =save_path_mean+f'/depth_{n_depth}_act_{activation}_mean.csv'
                    save_path_std=save_path_std+f'/depth_{n_depth}_act_{activation}_std.csv'
                    metrics_mean.to_csv(save_path_mean)
                    metrics_std.to_csv(save_path_std)

            for n_depth in depths:
                for activation in activations:
                    times_list=[]
                    metrics_list=[]
                    for rep in range(n_reps):
                        print(f"Running simulation qb {rep + 1}/{n_reps} with {n_person} persons and {n_item} items_with {n_depth} hidden_dim and {activation} activation")
                        # Load the data
                        res_mat = load_data(f'../data/n_p_{n_person}/n_i_{n_item}/rep_{rep}/res_mat.csv')
                        z_true = load_data(f'../data/n_p_{n_person}/n_i_{n_item}/rep_{rep}/z.csv')
                        a_true = load_data(f'../data/n_p_{n_person}/n_i_{n_item}/rep_{rep}/a.csv')
                        b_true = load_data(f'../data/n_p_{n_person}/n_i_{n_item}/rep_{rep}/b.csv')
                        # Initialize the model
                        model,start_time,end_time=train_model_qb(res_mat,n_item,n_person,n_depth,activation,device, epochs, early_stopping_count,batch_size,lr)
                        times_list,metrics_list=save_metrics_qb(start_time,end_time,times_list,metrics_list,model,res_mat,device,z_true,a_true,b_true,n_depth,activation,n_person,n_item,rep)
                    # Save the results
                    timese=np.array(times_list).mean()
                    total_time.append(timese)
                    metrics_mean=np.array(metrics_list).mean(axis=0)
                    metrics_mean=pd.DataFrame(metrics_mean,columns=['MAE','MSE','Bias'],index=['z','a','b'])
                    metrics_std=np.array(metrics_list).std(axis=0)
                    metrics_std=pd.DataFrame(metrics_std,columns=['MAE','MSE','Bias'],index=['z','a','b'])
                    save_path_mean=f'../metrics_summary/sim1/qb/n_p_{n_person}_ni_{n_item}'
                    save_path_std=f'../metrics_summary/sim1/qb/n_p_{n_person}_ni_{n_item}'
                    if not os.path.exists(save_path_mean):
                        os.makedirs(save_path_mean)
                    if not os.path.exists(save_path_std):
                        os.makedirs(save_path_std)
                    save_path_mean =save_path_mean+f'/depth_{n_depth}_act_{activation}_mean.csv'
                    save_path_std=save_path_std+f'/depth_{n_depth}_act_{activation}_std.csv'
                    metrics_mean.to_csv(save_path_mean)
                    metrics_std.to_csv(save_path_std)
            total_time=pd.DataFrame(total_time,columns=['time'])
            save_path_time=f'../metrics_summary/sim1/time'
            if not os.path.exists(save_path_time):
                os.makedirs(save_path_time)
            save_path_time=save_path_time+f'/n_p_{n_person}_n_i_{n_item}.csv'
            total_time.to_csv(save_path_time)

if __name__ == '__main__':
    args=parse_arguments()
    n_person_sim1=args.n_persons_sim
    n_item_sim1=args.n_items_sim
    n_reps=args.rep_sim
    depths=args.depths
    activations=args.activations
    batch_size=args.batch_size
    lr=args.lr
    epochs=args.epochs
    early_stopping_count=args.early_stopping_count
    run(n_person_sim1,n_item_sim1,n_reps,depths,activations,batch_size,lr,epochs,early_stopping_count)






















