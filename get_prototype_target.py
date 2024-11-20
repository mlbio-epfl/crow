import warnings
warnings.filterwarnings('ignore')

import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from utils import cluster_acc, accuracy
import dataset
import argparse
import os
import torch.nn.functional as F

import models
import torch.nn as nn
import torch.optim as optim
import numpy as np

import scanpy as sc
import anndata
import pandas as pd

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')

# 1. Set experiment parameteres
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--batch_size", type=int, default=1024)
args = parser.parse_args()

dataset_name = args.dataset
bs = args.batch_size

# For the clustering method, you can choose 'louvain' or 'k-means'
# In general, 'louvain' is more stable. However, it can be slow to run on large datasets.
# Here, we apply 'louvain' to 'Office' and 'OfficeHome', and 'k-means' to 'VisDA' and 'DomainNet'.
# Feel free to change the method :)

if dataset_name == 'Office':
    total_class_num = 31
    source = 'dslr'
    target_list = ['amazon', 'dslr', 'webcam']
    method = 'louvain'
elif dataset_name == 'OfficeHome':
    total_class_num = 65  
    source = 'Art'
    target_list = ['Art', 'Clipart', 'Product', 'RealWorld']
    method = 'louvain'
elif dataset_name == 'VisDA':
    total_class_num = 12
    source = 'validation'
    target_list = ['validation']
    method = 'k-means'
elif dataset_name == 'DomainNet':
    total_class_num = 345
    source = 'clipart'
    target_list = ['painting', 'real', 'sketch']
    method = 'k-means'
else:
    raise Exception('Invalid dataset name.')

# 2. Get centers
if method == 'k-means' or method == 'louvain':
    for target in target_list:

        # 2.1. Get features of all samples
        # get dataloader
        dataset_train = dataset.UniDA_lastlayerfeature(dataset_name, source, target)
        dataloader_train = DataLoader(dataset_train, batch_size=bs, shuffle=False, num_workers=32, pin_memory=True)

        # _t: target
        count = 0
        for _, _, _, feature_t, label_t, _ in dataloader_train:

            if count == 0:
                feature_t_all = feature_t
                label_t_all = label_t
            else:
                feature_t_all = torch.cat((feature_t_all, feature_t), dim = 0)
                label_t_all = torch.cat((label_t_all, label_t))

            count += 1

        feature_target = feature_t_all.to(torch.float32)
        feature_target = F.normalize(feature_target, dim=1)
        
        # 2.2. Get pseudo labels using clustering
        if method == 'k-means':
            kmeans = KMeans(n_clusters=total_class_num, n_init="auto").fit(feature_target.cpu())
            pred_t = kmeans.labels_

            unseen_acc_D2 = cluster_acc(pred_t, label_t_all.numpy())
            print(f'Accuracy of {dataset_name} {target}: {unseen_acc_D2}')

        else:
            feature_target_tmp = feature_target.numpy()
            dataframe = pd.DataFrame(feature_target_tmp)
            adata_ori = anndata.AnnData(dataframe)
            sc.pp.neighbors(adata_ori)

            num_cluster = 0
            res = 7.0
            rounds = 0
            step = 0.1

            while num_cluster != total_class_num:

                rounds += 1
                if rounds > 5:
                    step = step * 0.9
                    rounds = 0

                adata = adata_ori
                sc.tl.louvain(adata, resolution=res)

                pred_t = np.zeros(feature_target.shape[0])
                for i in range(feature_target.shape[0]):
                    pred_t[i] = adata.obs['louvain'][i]

                unseen_acc_D2=cluster_acc(pred_t, label_t_all.numpy())

                num_cluster = np.max(pred_t) + 1
                if num_cluster > total_class_num:
                    res -= step
                else:
                    res += step

            print(f'Accuracy of {dataset_name} {target}: {unseen_acc_D2}')


        # supervised
        model = models.MLP_simple(in_dim = 768, out_dim = int(num_cluster)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        pseudo_label = F.one_hot(torch.from_numpy(pred_t).to(torch.int64), num_classes=int(num_cluster)).to(device)

        for ep in range(1000):

            model.train()
            optimizer.zero_grad()

            feature_t = feature_target.to(torch.float32).to(device)

            output = model(feature_t)

            loss = criterion(output, pseudo_label.float())

            loss.backward()
            optimizer.step()

            # Test
            if (ep+1) % 100 == 0:

                model.eval()

                feature_t = feature_t.to(torch.float32).to(device)

                output = model(feature_t)
                pred_t_all = torch.argmax(output, dim = 1).cpu()

                acc_D2 = accuracy(pred_t_all.numpy(), pred_t)

                unseen_acc_D2 = cluster_acc(pred_t_all.numpy(), label_t_all.numpy())

        torch.save(model.state_dict()['head.weight'].cpu().T.numpy(), f'./ckpt/prototype_target/{dataset_name}_{target}.pth')
        print(f'Target prototype {dataset_name} {target} is saved.')

else:
    print('Wrong method name.')
