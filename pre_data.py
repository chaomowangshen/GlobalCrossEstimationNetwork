import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulations")
    parser.add_argument('--n_persons_sim', nargs='+', type=int, default=[100,500,1000])
    parser.add_argument('--n_items_sim', nargs='+', type=int, default=[10,20,30,40,50,60])
    parser.add_argument('--rep_sim', type=int, default=100)
    parser.add_argument('--basedir', type=str, default='data/sim1/')
    args = parser.parse_args()
    return args

def generate_data(n_persons, n_items, rep, basedir):
    for i in tqdm(range(rep)):
        # 为每个人生成能力参数z
        z = np.random.randn(n_persons, 1)
        z = np.clip(z, -3, 3)
        # 为每个项目生成区分参数a
        a = np.random.uniform(0.2, 3, (1, n_items))
        # 为每个项目生成难度参数b
        b = np.random.randn(1, n_items)
        b = np.clip(b, -3, 3)
        # 计算每个人-项目对的正面响应概率
        p = 1 / (1 + np.exp(-a * (z - b)))
        # 生成响应矩阵
        res_mat = np.random.binomial(1, p)

        # 创建目录结构
        dir_path = os.path.join(basedir, f"n_p_{n_persons}", f"n_i_{n_items}", f'rep_{i}')
        os.makedirs(dir_path, exist_ok=True)

        # 将数据保存为CSV文件
        pd.DataFrame(z).to_csv(os.path.join(dir_path, "z.csv"), index=False, header=False)
        pd.DataFrame(a).to_csv(os.path.join(dir_path, "a.csv"), index=False, header=False)
        pd.DataFrame(b).to_csv(os.path.join(dir_path, "b.csv"), index=False, header=False)
        pd.DataFrame(p).to_csv(os.path.join(dir_path, "p.csv"), index=False, header=False)
        pd.DataFrame(res_mat).to_csv(os.path.join(dir_path, "res_mat.csv"), index=False, header=False)



if __name__ == '__main__':
    args=parse_arguments()
    np.random.seed(123)
    for n_persons in args.n_persons_sim:
        for n_items in args.n_items_sim:
            generate_data(n_persons, n_items, args.rep_sim, args.basedir)
