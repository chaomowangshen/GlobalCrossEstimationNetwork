import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from time import time
from eve_metric import calculate_mae,calculate_mse,compute_bias
import os
import pandas as pd


#题目参数估计网络


    
#题目参数估计网络
class PersonNet(nn.Module):
    def __init__(self, num_items, depth=1,activation='sigmoid',hidden_dim=100):
        super(PersonNet, self).__init__()
        self.depth=depth
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_items, hidden_dim))
        for _ in range((self.depth - 1) // 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
            self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
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
        for layer in self.layers:
            x = self.activation(layer(x))
        x=self.linear(x)
        x=self.tanh(x)*3
        x = self.norm(x)
        return x

#项目参数估计网络
class ItemNet(nn.Module):
    def __init__(self, num_users, depth=1, activation='sigmoid', hidden_dim=100):
        super(ItemNet, self).__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_users, hidden_dim))
        for _ in range((self.depth - 1) // 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
            self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        self.linear_a = nn.Linear(hidden_dim, 1)
        self.linear_b = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        a = self.sig(self.linear_a(x)) * 3
        b = self.tanh(self.linear_b(x)) * 3
        return a, b  
    
    
    
    

#整合CEN模型
class CEN_QB(nn.Module):
    def __init__(self, inp_size_person_net, inp_size_item_net,depth=1,activation='sigmoid'):
        super(CEN_QB, self).__init__()
        self.person_net = PersonNet(inp_size_person_net, depth,activation)
        self.item_net = ItemNet(inp_size_item_net, depth,activation)
    def forward(self, x_person, x_item):
        z = self.person_net(x_person)
        a,b = self.item_net(x_item)
        a=a.reshape(1,-1)
        b=b.reshape(1,-1)
        p = 1 / (1 + torch.exp(-a * (z - b)))
        return p



def train_one_epoch_qb(model, res_mat, optimizer, device,loss_fn,batch_size):
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

def evaluate_qb(model, res_mat, device,z_true, a_true, b_true):
    res_mat = res_mat.clone().to(device)
    res_mat_item = res_mat.clone().T.to(device)
    z=model.person_net(res_mat).cpu()
    a,b=model.item_net(res_mat_item)
    a=a.cpu()
    b=b.cpu()
    #print(a,a_true)
    #print(b,b_true)
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


def save_metrics_qb(start_time,end_time,times_list,metrics_list,model,res_mat,device,z_true,a_true,b_true,depth,activation,n_person,n_item,rep,sim='sim1'):
    #print(sim)
    elapsed_time = end_time - start_time
    times_list.append(elapsed_time)
    # Evaluate the model
    model.load_state_dict(torch.load('best_model.pth'))
    metric0 = evaluate_qb(model, res_mat, device, z_true, a_true, b_true)
    metric0 = pd.DataFrame(metric0, columns=['MAE', 'MSE', 'Bias'], index=['z', 'a', 'b'])
    save_path = f'../metrics/{sim}/qb/n_p_{n_person}/ni_{n_item}/rep{rep}/hidden_dim{depth}/act{activation}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metric0.to_csv(f'{save_path}/metric{depth}_{activation}.csv')
    metrics_list.append(metric0)
    return times_list,metrics_list


def train_model_qb(res_mat,n_item,n_person,depth,activation,device, epochs, early_stopping_count,batch_size,lr=0.001):
    print(early_stopping_count,lr,depth)
    model=CEN_QB(n_item,n_person,depth,activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    early_stopping = EarlyStopping(patience=30, verbose=False)
    start_time = time()
    for epoch in range(epochs):
        loss = train_one_epoch_qb(model, res_mat, optimizer, device,loss_fn,batch_size)
        if np.isnan(loss):
            print("Loss is NaN. Stopping training.")
            break
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss}')
        early_stopping(loss, model)
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




