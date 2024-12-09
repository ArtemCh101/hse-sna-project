import torch.nn as nn
from dgl.nn import SAGEConv
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, dropout=False):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        if dropout:
            self.dropout = nn.Dropout(p=0.2)
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        if hasattr(self, 'dropout'):
            h = self.dropout(h)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h