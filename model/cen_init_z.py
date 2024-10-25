import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from time import time
import numpy as np
from eve_metric import calculate_mae,calculate_mse,compute_bias
import os
import pandas as pd
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#题目参数估计网络
class PersonNet(nn.Module):
    def __init__(self, num_items, depth=1,activation='sigmoid'):
        super(PersonNet, self).__init__()
        self.depth=depth
        fc1_dim=max(num_items//2,2)
        fc2_dim=max(num_items//4,2)
        fc3_dim=max(num_items//8,2)
        self.fc1 = nn.Linear(num_items, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc3_dim)
        if self.depth==1:
            self.linear=nn.Linear(fc1_dim,1)
        if self.depth==3:
            self.linear=nn.Linear(fc3_dim,1)
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
        x = self.linear(x)
        x = self.norm(x)
        return x

#项目参数估计网络
class ItemNet(nn.Module):
    def __init__(self):
        super(ItemNet, self).__init__()
    def get_X_item_net(self, ab):
        ab=ab
        return ab

#IRT模型估计网络
class IRTNet(nn.Module):
    def __init__(self):
        super(IRTNet, self).__init__()
    def forward(self, x):
        z = x[:, 0]
        a = x[:, 1]
        b = x[:, 2]
        p = 1 / (1 + torch.exp(-a * (z - b)))
        return p

#整合CEN模型
class CEN(nn.Module):
    def __init__(self, inp_size_person_net,depth=1,activation='sigmoid'):
        super(CEN, self).__init__()
        self.person_net = PersonNet(inp_size_person_net, depth,activation)
        self.item_net = ItemNet()
        self.irt_net = IRTNet()
    def forward(self, x_person, x_item):
        z = self.person_net(x_person)
        ab = self.item_net.get_X_item_net(x_item)
        x = torch.cat((z, ab), dim=1)
        p = self.irt_net(x)
        return p

    def preprocess_data(self, a_csv_path, b_csv_path,res_mat):
        n_persons, n_items = res_mat.shape
        # Get the input patterns 'X_person_net' for the person net.
        X_person_net = res_mat.repeat_interleave(n_items, dim=0)
        # Get the input patterns 'X_item_net' for the item net.
        a_true = pd.read_csv(a_csv_path,header=None).values.squeeze()
        b_true = pd.read_csv(b_csv_path,header=None).values.squeeze()
        X_item_net = np.vstack([a_true, b_true]).transpose()
        X_item_net = np.repeat(X_item_net[None, :], n_persons, axis=0)
        X_item_net = X_item_net.reshape((n_persons * n_items, 2))
        X_item_net=torch.tensor(X_item_net,dtype=torch.float32)
        # Get the the target patterns (labels) 'y_CEN' for CEN.
        y_CEN = res_mat.reshape(-1)
        return X_person_net, X_item_net, y_CEN


def train_one_epoch_init(model, a_csv_path, b_csv_path,res_mat, optimizer, device,loss_fn,batch_size):
    model.train()
    total_loss = 0
    X_person_net, X_item_net, y_CEN = model.preprocess_data(a_csv_path, b_csv_path,res_mat)
    dataset = TensorDataset(X_person_net, X_item_net, y_CEN)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for X_person, X_item, y in dataloader:
        X_person, X_item, y = X_person.to(device), X_item.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X_person, X_item)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

def evaluate_init(model, res_mat, device,z_true):
    res_mat=res_mat.to(device)
    z=model.person_net(res_mat).cpu()
    z= torch.clamp(z, min=-6, max=6)
    mae_z = calculate_mae(z, z_true)
    mse_z = calculate_mse(z, z_true)
    bias_z = compute_bias(z_true, z)
    return [[mae_z,mse_z,bias_z]]

def save_metrics(start_time,end_time,times_list,metrics_list,model,res_mat,device,z_true,depth,activation,n_person,n_item,rep,sim='sim2',res='res2_2'):
    elapsed_time = end_time - start_time
    times_list.append(elapsed_time)
    # Evaluate the model
    model.load_state_dict(torch.load('best_model_init_z.pth'))
    metric0 = evaluate_init(model, res_mat, device, z_true)
    metric0 = pd.DataFrame(metric0, columns=['MAE', 'MSE', 'Bias'], index=['z'])
    save_path = f'../metrics/{sim}/{res}/init/n_p_{n_person}/ni_{n_item}/rep{rep}/depth{depth}/act{activation}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metric0.to_csv(f'{save_path}/metric{depth}_{activation}.csv')
    metrics_list.append(metric0)
    return times_list,metrics_list

def train_model_init(res_mat,a_csv_path, b_csv_path,n_item,n_person,depth,activation,device, epochs, early_stopping_count,batch_size,lr=0.001):
    model=CEN(n_item,depth,activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(patience=early_stopping_count, verbose=False)
    start_time = time()
    for epoch in range(epochs):
        loss = train_one_epoch_init(model, a_csv_path, b_csv_path,res_mat, optimizer, device,loss_fn,batch_size)
        if np.isnan(loss):
            print("Loss is NaN. Stopping training.")
            break
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss}')
        early_stopping(loss, model,filename='best_model_init_z.pth')
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


