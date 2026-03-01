from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class NeighborSampler(torch.utils.data.DataLoader):
    r"""
    This code adapted from the pytorch geometric
    (https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/loader/neighbor_sampler.py).
    """
    def __init__(self,
                 edge_index: Union[Tensor, SparseTensor],
                 adj: SparseTensor,
                 sizes: List[int],
                 is_train: bool = False,
                 wt: int = 20,
                 wl: int = 4,
                 comm: Optional[Tensor] = None,
                 comm_list: Optional[List[List]] = None,
                 drop_last= False,
                 node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None,
                 return_e_id: bool = True,
                 transform: Callable = None,
                 theta: Optional[int] = None,
                 **kwargs):

        edge_index = edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        # Save for Pytorch Lightning < 1.6:
        self.edge_index = edge_index
        self.adj = adj
        self.node_idx = node_idx
        self.num_nodes = num_nodes
        self.is_train = is_train
        self.drop_last = drop_last
        self.wt = wt
        self.wl = wl
        self.theta = theta

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        self.comm = comm
        self.comm_list = comm_list
        
        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1
            value = torch.arange(edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, drop_last=self.drop_last, **kwargs)

    
    def get_batch(self, random_nodes):
        device = random_nodes.device
        batch_set = set(random_nodes.tolist())
        
        node_comm_ids = self.comm[random_nodes].long() 
        
        all_comm_sizes = torch.bincount(self.comm.long())
        seed_comm_sizes = all_comm_sizes[node_comm_ids]
        

        large_mask = seed_comm_sizes >= self.theta
        rw_seeds = random_nodes[large_mask]
        rw_seed_comms = node_comm_ids[large_mask].view(-1, 1) 
        if rw_seeds.numel() > 0:
            num_rw = rw_seeds.numel()
            seeds_rep = rw_seeds.repeat_interleave(self.wt)
            rw_res = self.adj.random_walk(seeds_rep, self.wl)[:, 1:] 
            rw_matrix = rw_res.reshape(num_rw, self.wt * self.wl)
            
            walk_node_comms = self.comm[rw_matrix]
            in_comm_mask = (walk_node_comms == rw_seed_comms)
            masked_rw_matrix = torch.where(in_comm_mask, rw_matrix, torch.tensor(-1, device=device))

            for i in range(num_rw):
                row_nodes = masked_rw_matrix[i]
                valid_row_nodes = row_nodes[row_nodes != -1]
                if valid_row_nodes.numel() > 0:
                    nodes, counts = torch.unique(valid_row_nodes, return_counts=True)
                    high_freq_mask = counts > counts.float().mean()
                    batch_set.update(nodes[high_freq_mask].tolist())


        small_comm_mask = (seed_comm_sizes < self.theta) & (seed_comm_sizes > 1)
        small_comm_seeds = random_nodes[small_comm_mask]
        if small_comm_seeds.numel() > 0:
            relevant_small_comms = torch.unique(node_comm_ids[small_comm_mask])
            for cid in relevant_small_comms.tolist():
                comm_nodes = (self.comm == cid).nonzero(as_tuple=False).view(-1)
                batch_set.update(comm_nodes.tolist())

        batch = torch.tensor(list(batch_set), device=device)
        

        batch_size = batch.shape[0]
        batch_repeat = batch.repeat(self.wt)
        rw2 = self.adj.random_walk(batch_repeat, self.wl)[:, 1:]
        if not isinstance(rw2, torch.Tensor):
            rw2 = rw2[0]
        rw2 = rw2.t().reshape(-1, batch_size).t()

        row, col, val = [], [], []
        for i in range(batch.shape[0]):
            rw2_nodes, rw2_times = torch.unique(rw2[i], return_counts=True)
            row += [batch[i].item()] * rw2_nodes.shape[0]
            col += rw2_nodes.tolist()
            val += rw2_times.tolist()

        unique_nodes = list(set(row + col))
        subg2g = dict(zip(unique_nodes, list(range(len(unique_nodes)))))

        row = [subg2g[x] for x in row]
        col = [subg2g[x] for x in col]
        idx = torch.tensor([subg2g[x] for x in batch.tolist()])

        adj_ = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), value=torch.tensor(val),
                            sparse_sizes=(len(unique_nodes), len(unique_nodes)))

        adj_batch, _ = adj_.saint_subgraph(idx)


        adj_batch_sp = adj_batch.to_scipy(layout='coo')
        adj_batch_sp.setdiag([0] * idx.shape[0])
        adj_batch = SparseTensor.from_scipy(adj_batch_sp)
        return batch, adj_batch

    

    
    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        adj_batch = None
        if self.is_train:
            batch, adj_batch = self.get_batch(batch)
        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out
        return out, adj_batch, batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'
