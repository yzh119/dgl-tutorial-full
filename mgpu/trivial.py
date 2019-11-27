import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
import time
import argparse
from dgl.nn.pytorch.conv import SAGEConv
from graph_io import load_graph

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
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

    csr = load_graph('er100000')
    g = dgl.DGLGraph(csr, readonly=True)
    if proc_id == 0:
        print(g)
    in_feats = 300
    n_classes = 10
    features = th.rand(g.number_of_nodes(), in_feats).to(dev_id)
    labels = th.randint(n_classes, (g.number_of_nodes(),)).to(dev_id)

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
    num_epochs = 10

    model = GraphSAGE(in_feats, n_hidden, n_classes, L, F.relu, dropout, 'gcn')
    model = model.to(dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    th.cuda.synchronize()

    avg = 0
    for epoch in range(num_epochs):
        i = 0
        elapsed_time = 0
        tic = time.time() 
        if n_gpus > 1:
            th.distributed.barrier()
        # forward
        pred = model(g, features)
        # compute loss
        loss = loss_fcn(pred, labels)
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
        toc = time.time()
        elapsed_time += toc - tic
        if n_gpus > 1:
            th.distributed.barrier()
        if proc_id == 0: print('epoch time: ', elapsed_time)
        if epoch >= 5:
            avg += elapsed_time
    
    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print('avg time: ', avg / (epoch - 4)) 

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
