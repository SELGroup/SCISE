from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning
from utils.util import cal_se_hard
from time import time
import torch
from torch_geometric.utils import degree
import numpy as np

def creat_comm(edge_index, N, save_path, args, draw_fig=False, cal_se=True):
    print('---------------------------------------------------')
    print('creating comms...')
    device = torch.device(args.device)
    ew = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    ew = ew.to(device)
    edges=  edge_index.to(device).t()
    deg = degree(edge_index[0], num_nodes=N)
    dist = deg * 2
    dist = dist.to(device)
    dist = dist / (2 * ew.sum())  # ew.sum()=vol(G) dist=di/vol(G)
    
    min_comm, p = args.commNum, args.p
    t0 = time()
    print(f'start encoding tree with p={p}, min_comm={min_comm}...')
    graph = GraphSparse(edges, ew, dist)
    optim = OperatorPropagation(Partitioning(graph, None))

    optim.perform(p=p, min_com=min_comm)
    division = optim.enc.node_id
    totol_comm = torch.max(division) + 1
    
    comms = {}
    comm = torch.zeros(N).to(device)
    for i in range(totol_comm):
        idx = division == i
        if idx.any():
            comm[idx] = i
            comms[i] = idx.nonzero().squeeze(1)

    print(f'finish encoding tree using {(time()-t0):.3f}s!')

    #store comms and comm
    checkpoint = {
        'comm': comm.cpu(), 
        'comms': {i: v.cpu() for i, v in comms.items()}, 
    }
    torch.save(checkpoint, save_path)
    print(f"comm division saved to {save_path}")

    if draw_fig:

        from collections import Counter
        import matplotlib.pyplot as plt
        community_sizes = [len(comms[i]) for i in range(totol_comm) if i in comms]

        size_counts = Counter(community_sizes)
        sorted_sizes = sorted(size_counts.keys())
        for size in sorted_sizes:
            count = size_counts[size]

        counts = [size_counts[s] for s in sorted_sizes]
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_sizes, counts, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of Community Sizes (p={p},comm={totol_comm})', fontsize=14)
        plt.xlabel('Number of Nodes in Community', fontsize=12)
        plt.ylabel('Number of Communities', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.yscale('log')
        plt.savefig(f"./figure/{args.dataset}/distribution/comm{totol_comm}_p{p}.png", bbox_inches='tight')

        bins = [1, 2, 3, 5, 10, 20, 50, 100, 300, 500, 1000, 2000, 3000, 5000, max(10000,max(community_sizes))]
        counts, _ = np.histogram(community_sizes, bins=bins)
        labels = [f"[{bins[i]}-{bins[i+1]})" for i in range(len(bins)-1)]
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(labels))
        plt.bar(x_pos, counts, width=0.8, color="#4C5BAF", edgecolor='black')
        plt.xticks(x_pos, labels, rotation=45)
        plt.title(f'Community Size Distribution by Groups (p={p},comm={totol_comm})', fontsize=14)
        plt.xlabel('Node Count Range', fontsize=12)
        plt.ylabel('Number of Communities', fontsize=12)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        for i in range(len(counts)):
            if counts[i] > 0:
                plt.text(i, counts[i], str(counts[i]), 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"./figure/{args.dataset}/bin/comm{totol_comm}_p{p}.png", bbox_inches='tight')
        
    if cal_se:    
        print('start cal SE...')
        t1 = time()
        _,SE2 = cal_se_hard(deg.to(device), comm.to(device), edge_index.to(device))  
        print(f'finish cal SE using {(time()-t1):.3f}s!')
        print(f'p={p},totol_comm={totol_comm}, se={SE2:.4f}')
        print('---------------------------------------------------')


    return checkpoint

    