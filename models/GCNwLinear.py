import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F

class GCNwLinear(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, dropout):
        super(GCNwLinear, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers

        # Layer containers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.linear = nn.ModuleList()
        
        # Input layer
        self.input_drop = nn.Dropout(min(0.1, dropout))
        self.convs.append(GraphConv(in_feats, h_feats, bias=False))
        self.bns.append(nn.BatchNorm1d(h_feats))
        self.linear.append(nn.Linear(in_feats, h_feats, bias=False))

        # Hidden layers 
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(h_feats, h_feats, bias=False))
            self.bns.append(nn.BatchNorm1d(h_feats))
            self.linear.append(nn.Linear(h_feats, h_feats, bias=False))
        
        # Output layer
        self.convs.append(GraphConv(h_feats, num_classes, bias=False))
        self.linear.append(nn.Linear(h_feats, num_classes, bias=False))

    def forward(self, g, in_feat):
        h = in_feat
        h = self.input_drop(h)

        for i in range(self.num_layers):
            c = self.convs[i](g, h)
            linear = self.linear[i](h)

            h = c + linear

            if i < self.num_layers - 1:
                h = self.bns[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h        