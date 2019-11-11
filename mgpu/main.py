import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
import time
import argparse
from dgl.data import RedditDataset


class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, nodes):
        h = nodes.data['h']
        h = self.linear(h)
        if self.activation is not None:
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
            h = nf.layers[i].data.pop('a')
            h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i, fn.copy_u('h', 'm'), fn.mean('m', 'h'), layer)

        h = nf.layers[-1].data.pop('a')
        return h


def run(proc_id, n_gpus, devices):
    th.manual_seed(1234)
    np.random.seed(1234)
    th.cuda.manual_seed_all(1234)

    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')

        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)

    data = RedditDataset(self_loop=True)
    train_nid = th.LongTensor(np.nonzero(data.train_mask)[0])
    features = th.Tensor(data.features)

    in_feats = features.shape[1]
    labels = th.LongTensor(data.labels).to(dev_id)
    n_classes = data.num_labels

    g = dgl.DGLGraph(data.graph, readonly=True)
    g.ndata['features'] = features.to(dev_id)

    # number of GCN layers
    L = 2
    # number of hidden units of a fully connected layer
    n_hidden = 64
    # dropout probability
    dropout = 0.2
    # batch size
    batch_size = 1000
    # number of neighbors to sample
    num_neighbors = 4
    # number of epochs
    num_epochs = 1

    model = GCNSampling(in_feats, n_hidden, n_classes, L, F.relu, dropout)
    model = model.to(dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    sampler = dgl.contrib.sampling.NeighborSampler(g, batch_size / n_gpus, num_neighbors,
                                                   neighbor_type='in', shuffle=True, num_hops=L, seed_nodes=train_nid)

    total_step = 128
    th.cuda.synchronize()
    for epoch in range(num_epochs):
        i = 0
        start = True
        for step, nf in enumerate(sampler):
            if start:
                if n_gpus > 1:
                    th.distributed.barrier()
                tic = time.time() 
                start = False
            nf.copy_from_parent()
            # forward
            pred = model(nf)
            batch_nids = nf.layer_parent_nid(-1).to(dev_id)
            batch_labels = labels[batch_nids]
            # compute loss
            loss = loss_fcn(pred, batch_labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            if n_gpus > 1:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        th.distributed.all_reduce(param.grad.data,
                                                  op=th.distributed.ReduceOp.SUM)
                        param.grad.data /= n_gpus
            optimizer.step()
            if step % 10 == 0 and proc_id == 0:
                print('epoch{} step {}: loss {}'.format(epoch, step, loss.item()))
            if step + 1 == total_step:
                break
        if n_gpus > 1:
            th.distributed.barrier()
        if proc_id == 0: print('epoch time: ', time.time() - tic)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0')
    args = argparser.parse_args()
    
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)
    if n_gpus == 1:
        run(0, n_gpus, devices)
    else:
        mp = th.multiprocessing
        mp.spawn(run, args=(n_gpus, devices), nprocs=n_gpus)
