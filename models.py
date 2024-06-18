import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_simple(nn.Module):
    '''
    A simple head to get the source prototypes
    '''
    def __init__(self, in_dim = 252, out_dim = 65):
        super(MLP_simple,self).__init__()
        self.head = NormedLinear(in_dim, out_dim)
        
    def forward(self,din):
        out = self.head(din)
        return out


class MLP_double(nn.Module):
    '''
    The head we use to identify seen classes and discover unseen classes
    '''
    def __init__(self, in_dim = 768, out_dim_1 = 35, out_dim_2 = 30):
        super(MLP_double,self).__init__()

        self.head_seen = NormedLinear(in_dim, out_dim_1)
        self.head_unseen = NormedLinear(in_dim, out_dim_2)
        self.in_dim = in_dim
        self.out_dim_2 = out_dim_2
        
    def forward(self,din):

        return torch.cat((self.head_seen(din), self.head_unseen(din)), dim = 1), self.head_seen(din), self.head_unseen(din)
    
    def init_head(self, checkpoint, num_seen):

            self.head_seen.weight.data = torch.from_numpy(checkpoint).T.float()[:, :num_seen]
            self.head_unseen.weight.data = torch.from_numpy(checkpoint).T.float()[:, num_seen:]
        
        return True
    
    def init_head_M(self, checkpoint, M, num_seen):

        tmp1 = self.head_seen.weight.data.clone().cpu().detach()
        tmp2 = self.head_unseen.weight.data.clone().cpu().detach()
        
        self.head_seen = NormedLinear(self.in_dim, num_seen)
        self.head_seen.weight.data = checkpoint['model_state_dict']['head.weight'][:, :num_seen]

        tmp = torch.cat((tmp1, tmp2), dim = 1)

        self.head_unseen = NormedLinear(self.in_dim, M.shape[1] - num_seen)
        
        for i in range(num_seen, M.shape[1]):
            pos = torch.argmax(M[:,i])
            self.head_unseen.weight.data[:, i - num_seen] = tmp[:, pos]

        return True

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out * 20


