import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from time import time
from eve_metric import calculate_mae,calculate_mse,compute_bias
import os
import pandas as pd
from data_loader import load_data


#题目参数估计网络
class PersonNet(nn.Module):
    def __init__(self, num_items, depth=1,activation='sigmoid',hidden_dim=100):
        super(PersonNet, self).__init__()
        self.depth=depth
        self.fc1=nn.Linear(num_items, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3=nn.Linear(hidden_dim*2, hidden_dim)
        self.linear=nn.Linear(hidden_dim,1)
        self.tanh = nn.Tanh()
        self.norm = nn.BatchNorm1d(1, eps=0, momentum=0, affine=False)
        if activation=='sigmoid':
            self.activation=nn.Sigmoid()
        elif activation=='relu':
            self.activation=nn.ReLU()
        elif activation=='leakyrelu':
            self.activation=nn.LeakyReLU()
        elif activation=='tanh':
            self.activation=nn.Tanh()
    def forward(self, x):
        if self.depth==1:
            x = self.activation(self.fc1(x))
        elif self.depth==3:
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.fc3(x))
        x=self.linear(x)
        x=self.tanh(x)*3
        x = self.norm(x)
        return x

#项目参数估计网络
class ItemNet(nn.Module):
    def __init__(self,a_csv_path,b_csv_path):
        super(ItemNet, self).__init__()
        self.a = load_data(a_csv_path)
        self.b = load_data(b_csv_path)
    def forward(self, device):
        a=self.a.to(device)
        b=self.b.to(device)
        return a,b


#整合CEN模型
class CEN_QB(nn.Module):
    def __init__(self,a_csv_path,b_csv_path,inp_size_person_net,depth=1,activation='sigmoid'):
        super(CEN_QB, self).__init__()
        self.person_net = PersonNet(inp_size_person_net, depth,activation)
        self.item_net = ItemNet(a_csv_path,b_csv_path)
    def forward(self, x_person, x_item):
        device=x_person.device
        z = self.person_net(x_person)
        a,b = self.item_net(device)
        a=a.reshape(1,-1)
        b=b.reshape(1,-1)
        p = 1 / (1 + torch.exp(-a * (z - b)))
        return p


def train_one_epoch_qb(model, res_mat, optimizer, device,loss_fn):
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

def evaluate_qb(model, res_mat, device,z_true):
    res_mat = res_mat.clone().to(device)
    z=model.person_net(res_mat).cpu()
    mae_z = calculate_mae(z.flatten(), z_true.flatten())
    mse_z = calculate_mse(z.flatten(), z_true.flatten())
    bias_z = compute_bias(z_true.flatten(), z.flatten())
    return [[mae_z,mse_z,bias_z]]


def save_metrics_qb(start_time,end_time,times_list,metrics_list,model,res_mat,device,z_true,depth,activation,n_person,n_item,rep,sim='sim2',res='res2_2'):
    elapsed_time = end_time - start_time
    times_list.append(elapsed_time)
    # Evaluate the model
    model.load_state_dict(torch.load('best_model_qb_z.pth'))
    metric0 = evaluate_qb(model, res_mat, device, z_true)
    metric0 = pd.DataFrame(metric0, columns=['MAE', 'MSE', 'Bias'], index=['z'])
    save_path = f'../metrics/{sim}/{res}/qb/n_p_{n_person}/ni_{n_item}/rep{rep}/hidden_dim{depth}/act{activation}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metric0.to_csv(f'{save_path}/metric{depth}_{activation}.csv')
    metrics_list.append(metric0)
    return times_list,metrics_list


def train_model_qb(res_mat,a_csv_path,b_csv_path,n_item,depth,activation,device, epochs, early_stopping_count,lr=0.001):
    model = CEN_QB(a_csv_path,b_csv_path,n_item,depth,activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    early_stopping = EarlyStopping(patience=early_stopping_count, verbose=False)
    start_time = time()
    for epoch in range(epochs):
        loss = train_one_epoch_qb(model, res_mat, optimizer, device,loss_fn)
        if np.isnan(loss):
            print("Loss is NaN. Stopping training.")
            break
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss}')
        early_stopping(loss, model,filename='best_model_qb_z.pth')
        if early_stopping.early_stop:
            print(f'Early stopping on epoch {epoch}')
            break
    end_time = time()
    return model,start_time,end_time

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




