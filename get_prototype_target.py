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

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')

# 1. Set experiment parameteres
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--batch_size", type=int, default=1024)
args = parser.parse_args()

dataset_name = args.dataset
bs = args.batch_size

if dataset_name == 'Office':
    total_class_num = 31
    source = 'dslr'
    target_list = ['amazon', 'dslr', 'webcam']
elif dataset_name == 'OfficeHome':
    total_class_num = 65  
    source = 'Art'
    target_list = ['Art', 'Clipart', 'Product', 'RealWorld']
elif dataset_name == 'VisDA':
    total_class_num = 12
    source = 'validation'
    target_list = ['validation']
elif dataset_name == 'DomainNet':
    total_class_num = 345
    source = 'clipart'
    target_list = ['painting', 'real', 'sketch']
else:
    raise Exception('Invalid dataset name.')

# 2. Get K-means centers
for target in target_list:

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

    kmeans = KMeans(n_clusters=total_class_num, n_init="auto").fit(feature_target.cpu())
    pred_t = kmeans.labels_

    unseen_acc_D2 = cluster_acc(pred_t, label_t_all.numpy())
    print(f'Accuracy of {dataset_name} {target}: {unseen_acc_D2}')

    # supervised learning using pseudo label get from K-means
    model = models.MLP_simple(in_dim = 768, out_dim = total_class_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    true_tmp = F.one_hot(torch.from_numpy(pred_t).to(torch.int64), num_classes=total_class_num).to(device)

    feature_t = feature_target.to(torch.float32).to(device)
    for ep in range(500):

        model.train()
        optimizer.zero_grad()

        output = model(feature_t)

        loss = criterion(output, true_tmp.float())

        loss.backward()
        optimizer.step()

    torch.save(model.state_dict()['head.weight'].cpu().T.numpy(), f'./ckpt/prototype_target/{dataset_name}_{target}.pth')
