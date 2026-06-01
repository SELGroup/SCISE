import pynvml
import os
import tracemalloc

def get_best_cuda_id():
    pynvml.nvmlInit()
    target_gpu = 1
    fallback_gpu = 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(target_gpu)
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        proc_count = len(procs)
    except pynvml.NVMLError:
        proc_count = 0
    pynvml.nvmlShutdown()
    return fallback_gpu if proc_count >= 1 else target_gpu

gpu_id = get_best_cuda_id()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print(f"Using physical GPU {gpu_id}")

import time
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit, Planetoid, Amazon, Coauthor
from torch_geometric.utils import to_undirected, add_remaining_self_loops, negative_sampling
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from model_gat import Model, Encoder
from utils.se_sampler_plus import NeighborSampler
from utils.util import setup_seed, get_mask, get_mask1, clustering, scale, z_score_scale, get_graph_stats, get_graph_stats
from utils.parser import parse_args
from utils.create_comm import creat_comm

import gc

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
#torch.cuda.set_per_process_memory_fraction(0.6, device=0)
#torch.cuda.set_per_process_memory_fraction(0.6, device=1)


def train(args, ts):
    tracemalloc.start()
    torch.cuda.reset_peak_memory_stats() 

    setup_seed(args.seed)
    # 逻辑 GPU 永远是 cuda:0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 立刻锁 GPU（创建 context）
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.empty(1, device=device)
    
    formatted_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(ts))
    log_file = f'./result/{args.dataset}/log/{formatted_time}.txt'
    save_file = f'./result/{args.dataset}/{formatted_time}.txt'
    print(f'test result saved at {save_file}')
    with open(save_file, 'w+') as file1:
        file1.write(f'{args}\n')
    print(args)
    if log_file is not None:
        with open(log_file, 'w+') as file_log:
            file_log.write(f'{args}\n')
    

    # 加载数据集
    if args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        path = './datasets/'
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(root=path, name=args.dataset)
        x, edge_index, y = dataset[0].x, dataset[0].edge_index, dataset[0].y
        y = y[:, 0]
    elif args.dataset == 'Reddit':
        path = './datasets/Reddit/'
        dataset = Reddit(root=path)
        x, edge_index, y = dataset[0].x, dataset[0].edge_index, dataset[0].y
    elif args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
        path = './datasets/'
        dataset = Planetoid(path, args.dataset)
        x, edge_index, y = dataset[0].x, dataset[0].edge_index, dataset[0].y
    elif args.dataset in ['Computers', 'Photo']:
        path = './datasets/'
        dataset = Amazon(path, args.dataset)
        x, edge_index, y = dataset[0].x, dataset[0].edge_index, dataset[0].y
    else:
        raise RuntimeError(f"Unknown dataset {args.dataset}")
    
    N, E, num_features = x.shape[0], edge_index.shape[-1], x.shape[-1]
    print(f"---------------------------------------------------")
    print(f"Loading {args.dataset} is over, num_nodes: {N: d}, num_edges: {E: d}, degree: {(E / N) :.5f}, "
          f"num_feats: {num_features: d}, time costs: {time.time()-ts: .2f}\n")
    if log_file is not None:
        with open(log_file, 'a') as file_log:
            file_log.write(f"---------------------------------------------------\n")
            file_log.write(f"Loading {args.dataset} is over, num_nodes: {N: d}, num_edges: {E: d}, degree: {(E / N) :.5f}, "
                           f"num_feats: {num_features: d}, time costs: {time.time()-ts: .2f}\n")
    
    #数据集加噪声
    if hasattr(args, 'noise_rate') and args.noise_rate != 0:        
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        if args.noise_rate > 0:
            # 1. 加噪声边 (Adding random noise edges)
            num_noise_edges = int(num_edges * args.noise_rate)
            # 进行负采样获取不存在的边作为噪声
            noise_edges = negative_sampling(
                edge_index=edge_index, 
                num_nodes=num_nodes, 
                num_neg_samples=num_noise_edges
            )
            edge_index = torch.cat([edge_index, noise_edges], dim=1)
            print(f"--- Added {num_noise_edges} noise edges (rate: {args.noise_rate}) ---")
            if log_file is not None:
                with open(log_file, 'a') as file_log:
                    file_log.write(f"--- Added {num_noise_edges} noise edges (rate: {args.noise_rate}) ---\n")
            
        elif args.noise_rate < 0:
            abs_rate = abs(args.noise_rate)
            row, col = edge_index
            
            # 1. 为每个有邻居的节点锁定一个“保底边”
            from torch_scatter import scatter_max
            edge_weight = torch.rand(num_edges, device=edge_index.device)
            _, keep_idx = scatter_max(edge_weight, row, dim=0, dim_size=num_nodes)
            
            # 过滤掉 -1 (无邻居节点) 以及 超过当前 edge 范围的索引 (边界溢出)
            keep_idx = keep_idx[(keep_idx != -1) & (keep_idx < num_edges)].unique()
            # ---------------------------
            
            # 2. 识别可动边
            all_indices = torch.arange(num_edges, device=edge_index.device)
            mask_removable = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
            mask_removable[keep_idx] = False
            removable_indices = all_indices[mask_removable]
            
            # 3. 计算实际要删除的数量
            num_to_remove = int(num_edges * abs_rate)
            num_to_remove = min(num_to_remove, len(removable_indices))
            
            # 4. 随机打乱并选择删除索引
            perm = torch.randperm(len(removable_indices), device=edge_index.device)
            remove_idx = removable_indices[perm[:num_to_remove]]
            
            # 5. 生成最终的边掩码
            final_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
            final_mask[remove_idx] = False
            edge_index = edge_index[:, final_mask]
            print(f"--- Removed {num_to_remove} edges (Rate: {args.noise_rate}) ---")
            if log_file is not None:
                with open(log_file, 'a') as file_log:
                    file_log.write(f"--- Removed {num_to_remove} edges (Rate: {args.noise_rate}) ---\n")


    # 处理数据集
    print('is undirected dataset:', dataset[0].is_undirected())
    if (not dataset[0].is_undirected()) and args.direct == 1:
        edge_index = add_remaining_self_loops(edge_index)[0]
        print('\n direct module!!!\n')
        if log_file is not None:
            with open(log_file, 'a') as file_log:
                file_log.write(f"direct module!!!\n")
    else:
        edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])
        print('\n undirect module!!!\n')
        if log_file is not None:
            with open(log_file, 'a') as file_log:
                file_log.write(f"undirect module!!!\n")

    N, E, num_features = x.shape[0], edge_index.shape[-1], x.shape[-1]
    print(f"Loading {args.dataset} is over, num_nodes: {N: d}, num_edges: {E: d}, degree: {(E / N) :.5f}, "
          f"num_feats: {num_features: d}, time costs: {time.time()-ts: .2f}")
    if log_file is not None:
        with open(log_file, 'a') as file_log:
            file_log.write(f"processing {args.dataset} is over, num_nodes: {N: d}, num_edges: {E: d}, degree: {(E / N) :.5f}, "
                           f"num_feats: {num_features: d}, time costs: {time.time()-ts: .2f}\n")
    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1], sparse_sizes=(N, N))
    adj.fill_value_(1.)

    hidden = list(map(int, args.hidden_channels.split(',')))
    if args.projection == '':
        projection = None
    else:
        projection = list(map(int, args.projection.split(',')))
    size = list(map(int, args.size.split(',')))
    assert len(hidden) == len(size)


    '''
    if args.dataset == 'Citeseer':
        for m in [1]:
            for p in [0.0005,0.005]:
                args.p = p
                args.commNum = m
                comm_data = creat_comm(edge_index=edge_index, N=N, args=args)
        exit()
    '''
    # 加载社区
    if hasattr(args, 'noise_rate') and args.noise_rate != 0: 
        comm_path = f"./comms/{args.dataset}/noise/division_comm{args.commNum}_p{args.p}_noise{args.noise_rate}.pt"
    else:
        comm_path = f"./comms/{args.dataset}/division_comm{args.commNum}_p{args.p}.pt"
    
    
    
    comm_path = f"./comms/{args.dataset}/time_test_{formatted_time}_division_comm{args.commNum}_p{args.p}.pt"
    if not os.path.exists(comm_path):
        print(f'---------------------------------------------------\ncreating comms...')
        print(f'start encoding tree with p={args.p}, min_comm={args.commNum}...')
        if log_file is not None:
            with open(log_file, 'a') as file_log:
                file_log.write(f'---------------------------------------------------\ncreating comms...\n')
                file_log.write(f'start encoding tree with p={args.p}, min_comm={args.commNum}...\n')
        
        t_se = time.time()
        comm_data = creat_comm(edge_index=edge_index, N=N, save_path=comm_path, args=args, draw_fig=False, cal_se=False)
        print(f'SE time: {(time.time()-t_se):.3f}s!')
        if log_file is not None:
            with open(log_file, 'a') as file_log:
                file_log.write(f'SE time: {(time.time()-t_se):.3f}s!\n')
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_memory = torch.cuda.max_memory_reserved() / (1024 ** 2)
        current, peak = tracemalloc.get_traced_memory()
        print(f"SE显存占用峰值: {peak_memory:.5f} MB, 分配峰值： {reserved_memory:.5f} MB, 内存占用峰值: {peak / (1024 ** 2):.5f} MB")
        if log_file is not None:
            with open(log_file, 'a') as file_log:
                file_log.write(f"SE显存占用峰值: {peak_memory:.5f} MB, 分配峰值： {reserved_memory:.5f} MB,内存占用峰值: {peak / (1024 ** 2):.5f} MB\n")
    else:
        comm_data = torch.load(comm_path)
    comm = comm_data['comm'].long().clone()
    comms = comm_data['comms']
    print(f"Loaded {len(comms)} communities.")
    if log_file is not None:
        with open(log_file, 'a') as file_log:
            file_log.write(f"Loaded {len(comms)} communities from {comm_path}.\n")

    #comm加随机噪声
    if hasattr(args, 'comm_noise_rate') and args.comm_noise_rate != 0:
        # 1. 计算需要加噪的节点数量
        num_noise_nodes = int(N * args.comm_noise_rate)
        num_comm = len(comms)
        # 2. 随机采样出需要修改标签的节点索引 (不重复采样)
        noise_node_indices = torch.randperm(N)[:num_noise_nodes]
        # 3. 随机社区
        comm[noise_node_indices] = torch.randint(0, num_comm, (num_noise_nodes,), device=comm.device)

        new_comms = {i: [] for i in range(num_comm)}
        for u, c in enumerate(comm.tolist()):
            new_comms[c].append(u)
        comms = {i: torch.tensor(v, dtype=torch.long).cpu() for i, v in new_comms.items()}
        print(f"--- Added {num_noise_nodes} noise comms-random (rate: {args.comm_noise_rate}) ---")
        if log_file is not None:
            with open(log_file, 'a') as file_log:
                file_log.write(f"--- Added {num_noise_nodes} noise comms (rate: {args.comm_noise_rate}) ---\n")
    
    #comm加邻居噪声
    elif hasattr(args, 'neighbor_noise_rate') and args.neighbor_noise_rate != 0:
        # 1. 计算需要加噪的节点数量
        num_noise_nodes = int(N * args.neighbor_noise_rate)
        num_comm = len(comms)
        # 2. 随机采样出需要修改标签的节点索引 (不重复采样)
        noise_node_indices = torch.randperm(N)[:num_noise_nodes]
        # 为了方便查找邻居，利用构建好的 adj (SparseTensor) 获取行指针和列索引
        # rowptr 可以让我们快速定位节点 i 的邻居在 col 中的起始和结束位置
        rowptr, col, _ = adj.csr()

        count_case_1 = 0  # 情况一：成功基于邻居社区加噪
        count_case_2 = 0  # 情况二：无邻居或邻居全同社区，触发全局保底


        # 3. 遍历每一个选中的加噪节点，将其修改为邻居的社区
        for node in noise_node_indices.tolist():
            current_comm = comm[node].item()  # 该节点原本的社区
            start, end = rowptr[node], rowptr[node + 1]
            neighbors = col[start:end]
            
            # 过滤出【和当前社区不一样】的邻居社区标签
            if len(neighbors) > 0:
                neighbor_comms = comm[neighbors]
                different_neighbor_comms = neighbor_comms[neighbor_comms != current_comm]
            else:
                different_neighbor_comms = torch.tensor([], dtype=torch.long, device=comm.device)
            
            # 核心策略判断：
            if len(different_neighbor_comms) > 0:
                # 情况一：存在和自己社区不同的邻居，从中随机挑一个
                random_idx = torch.randint(0, len(different_neighbor_comms), (1,)).item()
                new_c = different_neighbor_comms[random_idx].item()
                count_case_1 += 1
            else:
                # 情况二：没有邻居，或者所有邻居的社区都和自己一模一样
                # 保底策略：从全图【除当前社区外】的所有其他社区中随机选一个
                all_other_comms = [c for c in range(num_comm) if c != current_comm]
                if len(all_other_comms) > 0:
                    random_idx = torch.randint(0, len(all_other_comms), (1,)).item()
                    new_c = all_other_comms[random_idx]
                else:
                    # 极其极端的情况：全图总共就只有 1 个社区，无法变动
                    new_c = current_comm
                count_case_2 += 1
            
            comm[node] = new_c

        new_comms = {i: [] for i in range(num_comm)}
        for u, c in enumerate(comm.tolist()):
            new_comms[c].append(u)
        comms = {i: torch.tensor(v, dtype=torch.long).cpu() for i, v in new_comms.items()}
        print(f"--- Added {num_noise_nodes}(case1:{count_case_1}, case2:{count_case_2}) noise comms-neighbor (rate: {args.neighbor_noise_rate}) ---")
        if log_file is not None:
            with open(log_file, 'a') as file_log:
                file_log.write(f"--- Added {num_noise_nodes}(case1:{count_case_1}, case2:{count_case_2}) noise comms-neighbor (rate: {args.neighbor_noise_rate}) ---\n")
   

    train_loader = NeighborSampler(edge_index, adj,
                                   is_train=True,
                                   node_idx=None,
                                   wt=args.wt,
                                   wl=args.wl,
                                   theta=args.theta,
                                   comm = comm,
                                   comm_list = comms,
                                   sizes=size,
                                   batch_size=args.batchsize,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=8,
                                   pin_memory=True,
                                   prefetch_factor=4,
                                   persistent_workers=True)
    
    test_loader = NeighborSampler(edge_index, adj,
                                  is_train=False,
                                  node_idx=None,
                                  sizes=size,
                                  batch_size=30000,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=8,
                                  pin_memory=True,
                                  prefetch_factor=4,
                                  persistent_workers=True)

    encoder = Encoder(num_features, hidden_channels=hidden, base_model=GATConv, dropout=args.dropout, ns=args.ns).to(device)
    model = Model(encoder, in_channels=hidden[-1], project_hidden=projection, tau=args.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    dataset2n_clusters = {'ogbn-arxiv': 40, 'Reddit': 41,
                          'ogbn-products': 47, 'ogbn-papers100M': 172,
                          'Pubmed': 3, 'Computers': 10, 'Photo': 8, 'Citeseer': 6}
    n_clusters = dataset2n_clusters[args.dataset]
    #x = x.to(device)

    nmi_max = [0, 0, 0, 0, 0, 0] #nmi, ari, acc, f1, all. epoch
    all_max = [0, 0, 0, 0, 0, 0]
    #acc_max, ari_max, nmi_max, f1_max, all_metric_max, epoch_max = 0, 0, 0, 0, 0, 0
    step = 0
    print(f"---------------------------------------------------\nStart training")
    if log_file is not None:
        with open(log_file, 'a') as file_log:
            file_log.write(f"---------------------------------------------------\nStart training\n")
    
    for epoch in range(1, args.epochs):
        ts_epoch = time.time()
        # 强制清理垃圾
        gc.collect()
        torch.cuda.empty_cache()

        model.train()
        total_loss = total_examples = 0
        counter = 0
        ts_batch = time.time()

        #o_nodes, o_edges, o_degree = 0, 0, 0
        #e_nodes,e_edges,e_degree= 0, 0, 0
        for (batch_size, n_id, adjs), adj_batch, batch_ex, batch in train_loader:
            #print(f'examples: {batch_size:d}')
            # # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            '''
            orig_stats = get_graph_stats(batch, edge_index)
            ex_stats = get_graph_stats(batch_ex, edge_index)
            o_nodes += orig_stats["nodes"]
            o_edges += orig_stats["edges"]
            o_degree += orig_stats["avg_degree"]
            e_nodes += ex_stats["nodes"]
            e_edges += ex_stats["edges"]
            e_degree += ex_stats["avg_degree"]
            print(f"batch_original: nodes {orig_stats['nodes']}, edges {orig_stats['edges']}, avg_degree {orig_stats['avg_degree']}")
            print(f"batch_extend: nodes {ex_stats['nodes']}, edges {ex_stats['edges']}, avg_degree {ex_stats['avg_degree']}")
            '''
            if len(hidden) == 1:
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]

            adj_mask = get_mask1(adj_batch)
            optimizer.zero_grad()
            out = model(x[n_id].to(device), adjs=adjs)
            out = scale(out)
            out = F.normalize(out, p=2, dim=1)
            loss = model.loss(out, adj_mask)
            #loss = model.loss(out, adj_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.item())
            total_examples += batch_size

            if args.dataset in ['Pubmed','Computers','Photo','Citeseer']:
                verbose = 1
            else:
                verbose = 10
            if counter % verbose == 0:
                print(f'(Epoch {epoch:02d}) | Batch {counter:02d}, loss: {loss:.5f}, examples: {batch_size:d}, time: {(time.time()-ts_batch):.2f}s')
                if log_file is not None:
                    with open(log_file, 'a') as file_log:
                        file_log.write(f'(Epoch {epoch:02d}) | Batch {counter:02d}, loss: {loss:.5f}, examples: {batch_size:d}, time: {(time.time()-ts_batch):.2f}s\n')
                ts_batch = time.time()
                      #f"实际使用 (Allocated): {torch.cuda.memory_allocated() / 1024**2:.2f} MB",
                      #f"预留总额 (Reserved):  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            counter += 1
            # 显式删除本轮产生的庞大对象引用
            # del adj_mask, adjs, out, loss, adj_batch, batch

        print(f'------------------------------ epoch {epoch:02d} total loss:{total_loss:.5f}, time cost: {(time.time()-ts_epoch):.2f}s ------------------------------')
        #print(f"epoch_original: nodes {o_nodes/len(train_loader)}, edges {o_edges/len(train_loader)}, avg_degree {o_degree/len(train_loader)}")
        #print(f"epoch_extend: nodes {e_nodes/len(train_loader)}, edges {e_edges/len(train_loader)}, avg_degree {e_degree/len(train_loader)}")
        #exit()

        if log_file is not None:
            with open(log_file, 'a') as file_log:
                file_log.write(f'------------------------------ epoch {epoch:02d} total loss:{total_loss:.5f}, time cost: {(time.time()-ts_epoch):.2f}s ------------------------------\n')
        # 强制清理垃圾
        #gc.collect()
        #torch.cuda.empty_cache()
        if args.dataset in ['Pubmed','ogbn-products']:
            epoch_verbose = 1
            epoch_start = 1
        elif args.dataset in ['Computers']:
            epoch_verbose = 2
            epoch_start = 1
        elif args.dataset in ['Photo','Citeseer']:
            epoch_verbose = 10
            epoch_start = 200
        else:
            epoch_verbose = 5
            epoch_start = 30
        
        if (epoch >= epoch_start and epoch % epoch_verbose == 0) or (epoch == 1):
            ts_clustering = time.time()
            with torch.no_grad():
                model.eval()
                z = []
                for count, ((batch_size, n_id, adjs), _, _, _) in enumerate(tqdm(test_loader)):
                    if len(hidden) == 1:
                        adjs = [adjs]
                    adjs = [adj.to(device) for adj in adjs]
                    out = model(x[n_id].to(device), adjs=adjs)
                    z.append(out.detach().cpu().float())
                    # del adjs, out
                z = torch.cat(z, dim=0)
                z = scale(z)
                z = F.normalize(z, p=2, dim=1)
            
            print(f'Start clustering, num_clusters: {n_clusters: d}')
            if log_file is not None:
                with open(log_file, 'a') as file_log:
                    file_log.write(f'Start clustering, num_clusters: {n_clusters: d}\n')
            acc, nmi, ari, f1_macro, f1_micro = clustering(z, n_clusters, y.numpy(), kmeans_device=args.kmeans_device,
                                                        batch_size=args.kmeans_batch, tol=1e-4, device=device, spectral_clustering=False)
            # del z
            all_metric = acc + nmi + ari + f1_macro
            outline =f'Epoch {epoch:02d}, loss: {total_loss:.3f}, time cost: {time.time()-ts_epoch:.2f} s, nmi: {nmi:.5f}, ari: {ari:.5f}, acc: {acc:.5f}, f1_macro: {f1_macro:.5f}, all_metric: {all_metric:.5f}'
            
            # early stop
            improved = False
            if nmi > nmi_max[0]:
                nmi_max = [nmi, ari, acc, f1_macro, all_metric, epoch]
                improved = True
                outline = outline + f', new max NMI !!!'
            if all_metric > all_max[4]:
                all_max = [nmi, ari, acc, f1_macro, all_metric, epoch]
                improved = True
                outline = outline + f', new max ALL !!!'
                '''
                save_dir = './checkpoints/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                z_path = os.path.join(save_dir, f'{args.dataset}_embed.pt')
                torch.save(z, z_path)
                model_path = os.path.join(save_dir, f'{args.dataset}_model.pt')
                torch.save(model.state_dict(), model_path)
                print(f'---> Best embedding z (Epoch {epoch}) saved to {z_path}')
                '''
            if improved:
                step = 0
            else:
                step += 1
            outline = outline + f', step={step}'
            with open(save_file, 'a') as file1:
                file1.write(f'{outline}\n')
            print(f'{outline}, clustering time cost: {time.time() - ts_clustering:.2f}s')
            if log_file is not None:
                with open(log_file, 'a') as file_log:
                    file_log.write(f'{outline}, clustering time cost: {time.time() - ts_clustering:.2f}s\n')

            if (step > args.step and epoch > 30) or (time.time()-ts)/60/60 >= 2:
                outline = "*********************** early stop ***********************"
                with open(save_file, 'a') as file1:
                    file1.write(f'{outline}\n')
                print(outline)
                if log_file is not None:
                    with open(log_file, 'a') as file_log:
                        file_log.write(f'{outline}\n')
                break
        

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
        current, peak = tracemalloc.get_traced_memory()
        print(f"训练显存占用峰值: {peak_memory:.5f} GB, 分配峰值： {reserved_memory:.5f} GB, 内存占用峰值: {peak / (1024 ** 2):.5f} MB")
        if log_file is not None:
            with open(log_file, 'a') as file_log:
                file_log.write(f"训练显存占用峰值: {peak_memory:.5f} GB, 分配峰值： {reserved_memory:.5f} GB, 内存占用峰值: {peak / (1024 ** 2):.5f} MB\n")
    
    tracemalloc.stop()

    outline = f'best perform NMI at epoch{nmi_max[5]}, nmi: {nmi_max[0]:.5f}, ari: {nmi_max[1]:.5f}, acc: {nmi_max[2]:.5f}, f1_macro: {nmi_max[3]:.5f}, all_metric: {nmi_max[4]:.5f}'
    with open(save_file, 'a') as file1:
        file1.write(f'{outline}\n')
    print(outline)
    if log_file is not None:
        with open(log_file, 'a') as file_log:
            file_log.write(f'{outline}\n')
    outline = f'best perform ALL at epoch{all_max[5]}, nmi: {all_max[0]:.5f}, ari: {all_max[1]:.5f}, acc: {all_max[2]:.5f}, f1_macro: {all_max[3]:.5f}, all_metric: {all_max[4]:.5f}'
    with open(save_file, 'a') as file1:
        file1.write(f'{outline}\n')
    print(outline)
    print(f'test result saved at path: {save_file}')
    if log_file is not None:
        with open(log_file, 'a') as file_log:
            file_log.write(f'{outline}\n')
            file_log.write(f'test result saved at path: {save_file}\n')
    return nmi_max, all_max, args


if __name__ == '__main__':

    args = parse_args()
    args.dataset = 'Photo'
    if args.dataset == 'Photo':
        args.p = 0.1
        args.commNum = 300
        args.theta = 25
        args.wt = 120   #90
        args.wl = 2
        
        args.batchsize = 2048
        args.size = '10'
        args.hidden_channels = '1024'
        args.tau = 0.5
        args.ns = 0.3
        args.lr = 0.005
        args.step = 10
        args.epochs = 1001


    #path = f'./result/{args.dataset}/noise/comm_noise_SCISE.txt'
    path = f'./result/{args.dataset}/0_SCISE.txt'
    #args.commNum = 1
    for noise in [0]:
        #args.comm_noise_rate = noise
        NMI_list,ARI_list,ACC_list,F1_list,ALL_list=[],[],[],[],[]
        for i in range(1):
            #args.seed = random.randint(1, 1000)
            # 写入开始时间，方便区分不同批次的实验
            ts = time.time()
            timestamp = time.strftime('%Y%m%d %H:%M:%S', time.localtime(ts))
            nmi_max, all_max, args = train(args, ts)
            with open(path, 'a') as file:
                file.write(f'\n{"-"*25} test {i+1} {"-"*25}\nRun at: {timestamp}, total time cost = {(time.time()-ts)/60:.2f} min\n')
                #file.write(f'\n{"-"*50}\nRun at: {timestamp}, total time cost = {(time.time()-ts)/60:.2f} min\n')
                file.write(f'Args: {args}\n')
                result_line = f'        ARI MAX Result -> nmi: {nmi_max[0]:.5f}, ari: {nmi_max[1]:.5f}, acc: {nmi_max[2]:.5f}, f1_macro: {nmi_max[3]:.5f}, all: {nmi_max[4]:.5f}\n'
                file.write(result_line)
                result_line = f'        ALL MAX Result -> nmi: {all_max[0]:.5f}, ari: {all_max[1]:.5f}, acc: {all_max[2]:.5f}, f1_macro: {all_max[3]:.5f}, all: {all_max[4]:.5f}\n'
                file.write(result_line)
            NMI_list.append(all_max[0])
            ARI_list.append(all_max[1])
            ACC_list.append(all_max[2])
            F1_list.append(all_max[3])
            ALL_list.append(all_max[4])
        '''
        NMI_list,ARI_list,ACC_list,F1_list,ALL_list=np.array(NMI_list),np.array(ARI_list),np.array(ACC_list),np.array(F1_list),np.array(ALL_list)
        with open(path, 'a') as file:
            file.write(f'\n{"="*100}\n')
            avg_line = f'AVG | nmi: {NMI_list.mean():.5f}, ari: {ARI_list.mean():.5f}, acc: {ACC_list.mean():.5f}, f1: {F1_list.mean():.5f}, all: {ALL_list.mean():.5f}\n'
            file.write(avg_line)
            std_line = f'STD | nmi: {NMI_list.std():.5f}, ari: {ARI_list.std():.5f}, acc: {ACC_list.std():.5f}, f1: {F1_list.std():.5f}, all: {ALL_list.std():.5f}\n'
            file.write(std_line)
            file.write(f'{"="*100}\n\n\n')
        '''
        
