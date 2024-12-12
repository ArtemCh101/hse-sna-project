import torch.nn as nn
from dgl.nn import GraphConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, add_bn=True, dropout=0):
        """
        add_bn: Whether to add batch normalization layer after each graph convolution layer
        """
        super(GCN, self).__init__()
        
        self.add_bn = add_bn
        self.dropout = dropout
        
        #Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_feats, h_feats)) # in_feats -> h_feats
        if self.add_bn:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(h_feats))

        #Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(h_feats, h_feats)) # h_feats -> h_feats   
            if self.add_bn:
                self.bns.append(nn.BatchNorm1d(h_feats))
        
        #Output layer
        self.convs.append(GraphConv(h_feats, num_classes)) # h_feats -> num_classes
        
    def forward(self, g, in_feat):
        h = in_feat
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(g, h)
            if self.add_bn:
                h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](g, h)
        return h
    