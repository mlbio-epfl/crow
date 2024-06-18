import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dataset
import argparse
import os
import torch.nn.functional as F
import models

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')

# 1. Set experiment parameteres
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-2)
args = parser.parse_args()

dataset_name = args.dataset
bs = args.batch_size
epochs = args.epochs
lr = args.learning_rate

if dataset_name == 'Office':
    source = 'dslr'
    target_list = ['amazon', 'dslr', 'webcam']
    num_class = 31
elif dataset_name == 'OfficeHome':
    source = 'Art'
    target_list = ['Art', 'Clipart', 'Product', 'RealWorld']
    num_class = 65
elif dataset_name == 'VisDA':
    source = 'validation'
    target_list = ['validation']
    num_class = 12
elif dataset_name == 'DomainNet':
    source = 'clipart'
    target_list = ['painting', 'real', 'sketch']
    num_class = 345
else:
    raise Exception('Invalid dataset name.')

# 2. Get source prototypes by supervised learning
criterion = nn.CrossEntropyLoss()

for target in target_list:

    model = models.MLP_simple(in_dim = 768, out_dim = num_class).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # get dataloader
    dataset_train = dataset.UniDA_lastlayerfeature(dataset_name, source, target)
    dataloader_train = DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=32, pin_memory=True)

    for epoch in range(epochs):
        for _, _, _, feature_t, label_t, _ in dataloader_train:

            feature_t = feature_t.to(torch.float32).to(device)
            feature_t = F.normalize(feature_t, dim=1)

            label_t = F.one_hot(label_t, num_classes=num_class).to(device)

            model.train()
            optimizer.zero_grad()

            output = model(feature_t)

            loss = criterion(output, label_t.float())

            error = (torch.argmax(output, dim = 1) - torch.argmax(label_t, dim = 1))
            acc = (output.shape[0] - torch.count_nonzero(error)) / output.shape[0]
            
            # The training accuracy is just for a reference.
            print(f'Accuracy of {dataset_name} {target}: {acc.item()}')

            loss.backward()
            optimizer.step()

    torch.save({'model_state_dict': model.state_dict()}, f'./ckpt/prototype_source/{dataset_name}_{target}.pt')
