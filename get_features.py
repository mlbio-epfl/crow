import torch
import clip
import dataset
from torch.utils.data import DataLoader
import os
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Set experiment parameteres
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
args = parser.parse_args()

dataset_name = args.dataset

if dataset_name == 'Office':
    source = 'dslr'
    target_list = ['amazon', 'dslr', 'webcam']
elif dataset_name == 'OfficeHome':
    source = 'Art'
    target_list = ['Art', 'Clipart', 'Product', 'RealWorld']
elif dataset_name == 'VisDA':
    source = 'validation'
    target_list = ['validation', 'train']
elif dataset_name == 'DomainNet':
    source = 'clipart'
    target_list = ['painting', 'real', 'sketch']
else:
    raise Exception('Invalid dataset name.')

# 2. Load model
model, _ = clip.load("./ckpt/clip/ViT-L-14-336px.pt", device=device)

# 3. Save representation features
path_representations = f'./representations/{dataset_name}'
if not os.path.exists(path_representations):
    os.makedirs(path_representations)

with torch.no_grad():
    for target in target_list:
        path_representations = f'./representations/{dataset_name}/{target}'
        if not os.path.exists(path_representations):
            os.mkdir(path_representations)
        
        dataset_train = dataset.UniDA_dataset(dataset_name, source, target)
        dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=32, pin_memory=True)

        count = 0
        for img_s, label_s, img_t, label_t in dataloader_train:

            img_t = img_t.to(device)

            # compute features
            feature_lastlayer = model.encode_image(img_t)
            feature_lastlayer = feature_lastlayer.squeeze()

            # save featrues
            torch.save(feature_lastlayer.cpu(), f'{path_representations}/{count}.pt')
            count += 1