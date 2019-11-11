import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
from dgl.data import RedditDataset

data = RedditDataset(self_loop=True)
train_nid = th.nonzero(data.train_mask)
features = th.Tensor(data.features)

in_feats = features.shape[1]
labels = th.LongTensor(data.labels)
n_classes = data.num_labels

g = DGLGraph(data.graph, readonly=True)
g.ndata['features'] = features

# number of GCN layers
L = 2
# number of hidden units of a fully connected layer
n_hidden = 64

class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, nodes):
        h = nodes.data['h']
        h = self.linear(h)
        if self.activation:
            h = self.activation(h)
        return {'a': h} 

class GCNSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(GCNSampling, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            NodeUpdate(in_feats, n_hidden, activation))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                NodeUpdate(n_hidden, n_hidden, activation))
        # output layer
        self.layers.append(
            NodeUpdate(n_hidden, n_classes))

    def forward(self, nf):
        nf.layers[0].data['a'] = nf.layers[0].data['features']
        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i, fn.copy_u('h', 'm'), fn.mean('m', 'h'), layer)

        h = nf.layers[-1].data.pop('activation')
        return h

# dropout probability
dropout = 0.2
# batch size
batch_size = 1000
# number of neighbors to sample
num_neighbors = 4
# number of epochs
num_epochs = 1

model = GCNSampling(in_feats, n_hidden, n_classes, L, F.relu, dropout)
loss_fcn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)
sampler = dgl.contrib.sampling.NeighborSampler(g, batch_size, num_neighbors,
                                               neighbor_type='in', shuffle=True, num_hops=L, seed_nodes=train_nid)

for epoch in range(num_epochs):
    i = 0
    for step, nf in enumerate(sampler):
        nf.copy_from_parent()
        # forward
        pred = model(nf)
        batch_nids = nf.layer_parent_nid(-1)
        batch_labels = labels[batch_nids]
        # compute loss
        loss = loss_fcn(pred, batch_labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optim.step()
        print('step {}: loss {}'.format(step, loss.item())
        if step >= 32:
            break
