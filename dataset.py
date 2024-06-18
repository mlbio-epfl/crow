import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
from collections import defaultdict
import torch.nn.functional as F
import os
import clip
from torchvision import transforms

class UniDA_dataset(Dataset):

    '''
    This Dataset is to load the original images.
    '''

    def __init__(self, dataset_name, source, target):

        txt_path_source = f"./Datasets/{dataset_name}/{source}.txt"
        txt_path_target = f"./Datasets/{dataset_name}/{target}.txt"
        path_prefix = f"./Datasets/{dataset_name}/"

        with open(txt_path_source) as f:
            lst = []
            for ind, x in enumerate(f.readlines()):
                
                item = {}
                impath = x.split(' ')[0]
                if dataset_name == 'Office':
                    tmp = impath.split('/')
                    impath = os.path.join(path_prefix, tmp[1], tmp[3], tmp[4])
                elif dataset_name == 'OfficeHome':
                    impath = os.path.join(path_prefix, impath[5:])
                elif dataset_name == 'VisDA':
                    impath = os.path.join(path_prefix, source, impath)
                elif dataset_name == 'DomainNet':
                    impath = os.path.join(path_prefix, impath)
                label = x.split(' ')[1].strip()
                classname = impath.split('/')[-2].replace('_', ' ')
                item = {'impath': impath,
                        'label': int(label),
                        'classname': classname}
                lst.append(item)

        self.images_source = lst

        with open(txt_path_target) as f:
            lst = []
            for ind, x in enumerate(f.readlines()):
                item = {}
                impath = x.split(' ')[0]
                if dataset_name == 'Office':
                    tmp = impath.split('/')
                    impath = os.path.join(path_prefix, tmp[1], tmp[3], tmp[4])
                elif dataset_name == 'OfficeHome':
                    impath = os.path.join(path_prefix, impath[5:])
                elif dataset_name == 'VisDA':
                    impath = os.path.join(path_prefix, target, impath)
                elif dataset_name == 'DomainNet':
                    impath = os.path.join(path_prefix, impath)
                label = x.split(' ')[1].strip()
                classname = impath.split('/')[-2].replace('_', ' ')
                item = {'impath': impath,
                        'label': int(label),
                        'classname': classname}
                lst.append(item)

        self.images_target = lst

        _, self.preprocess = clip.load("./ckpt/ViT-L-14-336px.pt")

    def __getitem__(self, idx):

        img_s = self.preprocess(Image.open(self.images_source[idx % len(self.images_source)]['impath']))
        label_s = self.images_source[idx % len(self.images_source)]['label']

        img_t = self.preprocess(Image.open(self.images_target[idx % len(self.images_target)]['impath']))
        label_t = self.images_target[idx % len(self.images_target)]['label']

        return img_s, label_s, img_t, label_t
    
    def __len__(self):
        return max(len(self.images_source), len(self.images_target))


class UniDA_lastlayerfeature(Dataset):

    '''
    This Dataset is to load the precomputed features.
    '''

    def __init__(self, name, source, target):

        self.dataset_name = name

        self.source = source
        self.target = target

        txt_path_source = f"./Datasets/{name}/{source}.txt"
        txt_path_target = f"./Datasets/{name}/{target}.txt"
        path_prefix = f"./Datasets/{name}/"

        with open(txt_path_source) as f:
            lst = []
            for ind, x in enumerate(f.readlines()):
                #print(ind, x)
                item = {}
                impath = x.split(' ')[0]
                if name == 'Office':
                    tmp = impath.split('/')
                    impath = os.path.join(path_prefix, tmp[1], tmp[3], tmp[4])
                elif name == 'OfficeHome':
                    impath = os.path.join(path_prefix, impath[5:])
                elif name == 'VisDA':
                    impath = os.path.join(path_prefix, source, impath)
                elif name == 'DomainNet':
                    impath = os.path.join(path_prefix, impath)
                else:
                    impath = os.path.join(path_prefix, impath)
                
                label = x.split(' ')[1].strip()
                classname = impath.split('/')[-2].replace('_', ' ')
                item = {'impath': impath,
                        'label': int(label),
                        'classname': classname}
                
                lst.append(item)

        self.images_source = lst

        with open(txt_path_target) as f:
            lst = []
            for ind, x in enumerate(f.readlines()):
                item = {}
                impath = x.split(' ')[0]
                if name == 'Office':
                    tmp = impath.split('/')
                    impath = os.path.join(path_prefix, tmp[1], tmp[3], tmp[4])
                elif name == 'OfficeHome':
                    impath = os.path.join(path_prefix, impath[5:])
                elif name == 'VisDA':
                    impath = os.path.join(path_prefix, target, impath)
                elif name == 'DomainNet':
                    impath = os.path.join(path_prefix, impath)
                else:
                    impath = os.path.join(path_prefix, impath)
                
                label = x.split(' ')[1].strip()
                classname = impath.split('/')[-2].replace('_', ' ')
                item = {'impath': impath,
                        'label': int(label),
                        'classname': classname}
                lst.append(item)

        self.images_target = lst

    def __getitem__(self, idx):

        idx_s = idx % len(self.images_source)
        feature_s = torch.load(f'./representations/{self.dataset_name}/{self.source}/{idx_s}.pt')
        label_s = self.images_source[idx_s]['label']

        idx_t = idx % len(self.images_target)
        feature_t = torch.load(f'./representations/{self.dataset_name}/{self.target}/{idx_s}.pt')

        label_t = self.images_target[idx_t]['label']

        feature_s = torch.autograd.Variable(feature_s,requires_grad = False)
        feature_t = torch.autograd.Variable(feature_t,requires_grad = False)

        return feature_s, label_s, idx_s, feature_t, label_t, idx_t
    
    def __len__(self):
        return max(len(self.images_source), len(self.images_target))