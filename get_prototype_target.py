import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from utils import cluster_acc
import dataset
import argparse
import os
import torch.nn.functional as F

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

    kmeans = KMeans(n_clusters=total_class_num, random_state=8, n_init="auto").fit(feature_target.cpu())
    pred_t = kmeans.labels_

    centers = kmeans.cluster_centers_
    torch.save(centers, f'./ckpt/prototype_target/{dataset_name}_{target}.pth')

    unseen_acc_D2 = cluster_acc(pred_t, label_t_all.numpy())
    print(f'Accuracy of {dataset_name} {target}: {unseen_acc_D2}')