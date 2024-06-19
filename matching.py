import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
import argparse

import models
from utils import accuracy, cluster_acc_2, entropy
import dataset
from torch.utils.data import DataLoader

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')

# 1. Set experiment parameteres
parser = argparse.ArgumentParser()

parser.add_argument("--dataset")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seen_classes", type=float, default=0.5)

parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--threshold", type=float, default=0.3)
parser.add_argument("--weight_reg", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=0.01)
parser.add_argument("--iteration_max", type=int, default=1000)
parser.add_argument("--iteration_eva", type=int, default=50)

args = parser.parse_args()

dataset_name = args.dataset
bs = args.batch_size
seen_classes = args.seen_classes

lr = args.learning_rate
threshold = args.threshold
weight_reg = args.weight_reg
tau = args.temperature
iteration_max = args.iteration_max
iteration_eva = args.iteration_eva

# If you want to freeze the CLIP ViT, change this to False
finetune = True

if dataset_name == 'Office':
    source_list = ['amazon', 'dslr', 'webcam']
    target_list = ['amazon', 'dslr', 'webcam']
    total_num = 31
elif dataset_name == 'OfficeHome':
    source_list = ['Art', 'Clipart', 'Product', 'RealWorld']
    target_list = ['Art', 'Clipart', 'Product', 'RealWorld']
    total_num = 65
if dataset_name == 'VisDA':
    source_list = ['train']
    target_list = ['validation']
    total_num = 12
elif dataset_name == 'DomainNet':
    source_list = ['painting', 'real', 'sketch']
    target_list = ['painting', 'real', 'sketch']
    total_num = 345

seen_num = int(round(total_num * seen_classes))

criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=0)

# Go through all the source and target pairs
for source in source_list:
    for target in target_list:
        
        # 2. Initial for the experiment
        # 2.0. skip when source and target from the same domain
        if source == target:
            continue

        print('{} to {}'.format(source, target))

        # 2.1. Get dataloader
        dataset_train = dataset.UniDA_dataset(dataset_name, source, target)
        dataloader_train = DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=64, pin_memory=True)

        dataset_val = dataset.UniDA_dataset(dataset_name, target, target)
        dataloader_val = DataLoader(dataset_val, batch_size=bs, shuffle=True, num_workers=64, pin_memory=True)

        # 2.2. Get head
        # 2.2.1. Get feature extractor
        feature_extractor, _ = clip.load("./ckpt/clip/ViT-L-14-336px.pt", device=device)
        for name, param in feature_extractor.named_parameters(): 
            param.requires_grad = False

        if finetune == True:
            for name, param in feature_extractor.named_parameters(): 
                # You can try to change the layers to finetuen!
                if 'visual.proj' in name or 'resblocks.23' in name or 'resblocks.22' in name or 'ln_post' in name:
                    param.requires_grad = True

        # 2.2.2. Get head (classifier)
        head = models.MLP_double(in_dim = 768, out_dim_1 = seen_num, out_dim_2 = total_num - seen_num)

        checkpoint = torch.load(f'./ckpt/prototype_target/{dataset_name}_{target}.pth', map_location=device)
        head.init_head(checkpoint, seen_num)

        head = head.to(device)
        
        # 3. Matching
        # 3.1. Get the distribution matrix Gamma
        Gamma = torch.zeros(total_num, seen_num).to(device)
        
        for img_s, label_s, _, _ in dataloader_train:

            feature_extractor.eval()
            head.eval()

            with torch.no_grad():

                img_s = img_s.to(device)

                feature_s = feature_extractor.encode_image(img_s).to(torch.float32)

                mask = label_s < seen_num
                feature_s = feature_s[mask]
                label_s = label_s[mask]

                # If the seen_num is very small, all of the samples in the batch can be masked out
                if feature_s.shape[0] == 0:
                    continue
                
                feature_s = feature_s.to(torch.float32).to(device)
                output_tmp, _, _ = head(feature_s)
                
                for j in range(label_s.shape[0]):
                    pred_tmp = torch.argmax(output_tmp[j]).cpu().numpy()
                    Gamma[pred_tmp, label_s[j]] += 1
        
        # 3.2. Get the distribution matrix D (Equation 1 in paper)
        D = softmax(Gamma / tau)

        # 3.3. Get the matching matrix M (Equation 2 in paper)
        D_tmp = torch.round(D + 0.5 - threshold)

        for j in range(total_num):
            if torch.sum(D_tmp[j,:]) < 0.1:
                tmp = torch.zeros(total_num,1).to(device)
                tmp[j,0] += 1
                D = torch.cat((D, tmp), dim = 1)

        M = torch.round(D + 0.5 - threshold)

        # 3.4. Initailize the head (W in paper) after matching
        checkpoint_seen = torch.load(f'./ckpt/prototype_source/{dataset_name}_{source}.pt')
        head.init_head_M(checkpoint_seen, M, seen_num)
        head = head.to(device)

        # 3.5. Set the optimizer
        optimizer_fc = optim.SGD(feature_extractor.parameters(), lr = lr * 0.1)
        parameters_set = []
        parameters_set.extend(head.head_seen.parameters())
        optimizer_cls = optim.SGD(parameters_set, lr = lr)

        # 4. Finetuning: note that we train the head by iteration
        iteration_num_total = 0
        for ep in range(int(iteration_max / iteration_eva)):

            # 4.1. Finetuning by iteration
            iteration_num = 0
            for img_s, label_s, img_t, label_t in dataloader_train:
                mask = label_s < seen_num
                img_s = img_s[mask]
                label_s = label_s[mask]

                if img_s.shape[0] == 0:
                    continue

                feature_extractor.train()
                head.train()

                optimizer_fc.zero_grad()
                optimizer_cls.zero_grad()

                img_s = img_s.to(device)
                img_t = img_t.to(device)

                feature_s = feature_extractor.encode_image(img_s).to(torch.float32)
                feature_t = feature_extractor.encode_image(img_t).to(torch.float32)

                output_source, _, _ = head(feature_s)
                output_target, _, _ = head(feature_t)

                label_s = F.one_hot(label_s.to(torch.int64), num_classes = output_source.shape[1]).to(device)

                # Get loss reg
                reg = -entropy(torch.mean(output_target, 0))

                loss = criterion(output_source, label_s.float()) + reg * weight_reg

                loss.backward(retain_graph=True)
                
                optimizer_fc.step()
                optimizer_cls.step()
                
                iteration_num += 1
                iteration_num_total += 1

                if iteration_num == iteration_max / 50:
                    break

            # 4.2. Evaluation after each number of 'iteration_eva' iterations
            with torch.no_grad():

                count_tmp = 0
                for _, _, img_t, label_t in dataloader_val:

                    feature_extractor.eval()
                    head.eval()

                    img_t = img_t.to(device)
                    feature_t = feature_extractor.encode_image(img_t).to(torch.float32)
                    output_t, _, _ = head(feature_t)

                    if count_tmp == 0:
                        output_target = output_t
                        true_target = label_t
                        feature_target = feature_t
                        count_tmp += 1
                    else:
                        output_target = torch.cat((output_target, output_t), dim = 0)
                        feature_target = torch.cat((feature_target, feature_t), dim = 0)
                        true_target = torch.cat((true_target, label_t), dim = 0)
                        count_tmp += 1

            output_target_test = output_target
            pred_target = torch.argmax(output_target_test, dim = 1).cpu().numpy()
            seen_mask_true_target = true_target < seen_num

            # 2.1. Seen accuracy
            seen_acc_target = accuracy(pred_target[seen_mask_true_target], true_target[seen_mask_true_target].numpy())

            # 2.2. Unseen accuracy
            unseen_mask_true_target = ~seen_mask_true_target
            unseen_acc_target = cluster_acc_2(pred_target[unseen_mask_true_target], true_target[unseen_mask_true_target].numpy(), seen_num-1)
            
            print('Iter {}: Seen accuracy - {:.3f}; Unseen accuracy - {:.3f}'.format(iteration_num_total, seen_acc_target, unseen_acc_target))
