import pynvml
import os
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
from utils.util import setup_seed, get_mask, get_mask1, clustering, scale, z_score_scale
from utils.parser import parse_args
from utils.create_comm import creat_comm

import gc




def train(args, ts):
    setup_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.empty(1, device=device)
    
    formatted_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(ts))
    save_file = f'./result/{args.dataset}/{formatted_time}.txt'
    print(f'test result saved at {save_file}')
    with open(save_file, 'w+') as file1:
        file1.write(f'{args}\n')
    print(args)
    

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
    elif args.dataset in ["CS", "Physics"]:
        path = './datasets/'
        dataset = Coauthor(path, args.dataset)
        x, edge_index, y = dataset[0].x, dataset[0].edge_index, dataset[0].y

    else:
        raise RuntimeError(f"Unknown dataset {args.dataset}")
    
    if hasattr(args, 'noise_rate') and args.noise_rate != 0:        
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        if args.noise_rate > 0:
            num_noise_edges = int(num_edges * args.noise_rate)
            noise_edges = negative_sampling(
                edge_index=edge_index, 
                num_nodes=num_nodes, 
                num_neg_samples=num_noise_edges
            )
            edge_index = torch.cat([edge_index, noise_edges], dim=1)
            print(f"--- Added {num_noise_edges} noise edges (rate: {args.noise_rate}) ---")
            
        elif args.noise_rate < 0:
            abs_rate = abs(args.noise_rate)
            row, col = edge_index

            from torch_scatter import scatter_max
            edge_weight = torch.rand(num_edges, device=edge_index.device)
            _, keep_idx = scatter_max(edge_weight, row, dim=0, dim_size=num_nodes)
   
            keep_idx = keep_idx[(keep_idx != -1) & (keep_idx < num_edges)].unique()
            all_indices = torch.arange(num_edges, device=edge_index.device)
            mask_removable = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
            mask_removable[keep_idx] = False
            removable_indices = all_indices[mask_removable]

            num_to_remove = int(num_edges * abs_rate)
            num_to_remove = min(num_to_remove, len(removable_indices))

            perm = torch.randperm(len(removable_indices), device=edge_index.device)
            remove_idx = removable_indices[perm[:num_to_remove]]

            final_mask = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
            final_mask[remove_idx] = False
            edge_index = edge_index[:, final_mask]
            print(f"--- Removed {num_to_remove} edges (Rate: {args.noise_rate}) ---")


    edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])

    N, E, num_features = x.shape[0], edge_index.shape[-1], x.shape[-1]
    print(f"Loading {args.dataset} is over, num_nodes: {N: d}, num_edges: {E: d}, "
          f"num_feats: {num_features: d}, time costs: {time.time()-ts: .2f}")
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

    comm_path = f"./comms/{args.dataset}/division_comm{args.commNum}_p{args.p}.pt"
    if not os.path.exists(comm_path):
        comm_data = creat_comm(edge_index=edge_index, N=N, save_path=comm_path, args=args, draw_fig=False)
    else:
        comm_data = torch.load(comm_path)
    comm = comm_data['comm']
    comms = comm_data['comms']
    print(f"Loaded {len(comms)} communities.")
    

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
 

    nmi_max = [0, 0, 0, 0, 0, 0] 
    all_max = [0, 0, 0, 0, 0, 0]

    step = 0
    print(f"Start training")
    for epoch in range(1, args.epochs):
        ts_epoch = time.time()
        gc.collect()
        torch.cuda.empty_cache()

        model.train()
        total_loss = total_examples = 0
        counter = 0
        ts_batch = time.time()
        for (batch_size, n_id, adjs), adj_batch, batch in train_loader:
            if len(hidden) == 1:
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]

            adj_mask = get_mask1(adj_batch)
            optimizer.zero_grad()
            out = model(x[n_id].to(device), adjs=adjs)
            out = scale(out)
            out = F.normalize(out, p=2, dim=1)
            loss = model.loss(out, adj_mask)
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
                ts_batch = time.time()
            counter += 1


        print(f'------------------------------ epoch {epoch:02d} total loss:{total_loss:.5f}, time cost: {(time.time()-ts_epoch):.2f}s ------------------------------')
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
            with torch.no_grad():
                model.eval()
                z = []
                for count, ((batch_size, n_id, adjs), _, batch) in enumerate(tqdm(test_loader)):
                    if len(hidden) == 1:
                        adjs = [adjs]
                    adjs = [adj.to(device) for adj in adjs]
                    out = model(x[n_id].to(device), adjs=adjs)
                    z.append(out.detach().cpu().float())
                z = torch.cat(z, dim=0)
                z = scale(z)
                z = F.normalize(z, p=2, dim=1)

            ts_clustering = time.time()
            print(f'Start clustering, num_clusters: {n_clusters: d}')
            acc, nmi, ari, f1_macro, f1_micro = clustering(z, n_clusters, y.numpy(), kmeans_device=args.kmeans_device,
                                                        batch_size=args.kmeans_batch, tol=1e-4, device=device, spectral_clustering=False)
            all_metric = acc + nmi + ari + f1_macro
            outline =f'Epoch {epoch:02d}, loss: {total_loss:.3f}, time cost: {(time.time()-ts_epoch)/60:.2f} min, nmi: {nmi:.5f}, ari: {ari:.5f}, acc: {acc:.5f}, f1_macro: {f1_macro:.5f}, all_metric: {all_metric:.5f}'
            
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

            
            if (step > args.step and epoch > 30) or (time.time()-ts)/60/60 >= 2:
                outline = "*********************** early stop ***********************"
                with open(save_file, 'a') as file1:
                    file1.write(f'{outline}\n')
                print(outline)
                break

    outline = f'best perform NMI at epoch{nmi_max[5]}, nmi: {nmi_max[0]:.5f}, ari: {nmi_max[1]:.5f}, acc: {nmi_max[2]:.5f}, f1_macro: {nmi_max[3]:.5f}, all_metric: {nmi_max[4]:.5f}'
    with open(save_file, 'a') as file1:
        file1.write(f'{outline}\n')
    print(outline)
    outline = f'best perform ALL at epoch{all_max[5]}, nmi: {all_max[0]:.5f}, ari: {all_max[1]:.5f}, acc: {all_max[2]:.5f}, f1_macro: {all_max[3]:.5f}, all_metric: {all_max[4]:.5f}'
    with open(save_file, 'a') as file1:
        file1.write(f'{outline}\n')
    print(outline)
    print(f'test result saved at path: {save_file}')
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


    path = f'./result/{args.dataset}/noise/SCISE.txt'

    NMI_list,ARI_list,ACC_list,F1_list,ALL_list=[],[],[],[],[]
    for i in range(3):
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
    
    NMI_list,ARI_list,ACC_list,F1_list,ALL_list=np.array(NMI_list),np.array(ARI_list),np.array(ACC_list),np.array(F1_list),np.array(ALL_list)
    with open(path, 'a') as file:
        file.write(f'\n{"="*100}\n')
        avg_line = f'AVG | nmi: {NMI_list.mean():.5f}, ari: {ARI_list.mean():.5f}, acc: {ACC_list.mean():.5f}, f1: {F1_list.mean():.5f}, all: {ALL_list.mean():.5f}\n'
        file.write(avg_line)
        std_line = f'STD | nmi: {NMI_list.std():.5f}, ari: {ARI_list.std():.5f}, acc: {ACC_list.std():.5f}, f1: {F1_list.std():.5f}, all: {ALL_list.std():.5f}\n'
        file.write(std_line)
        file.write(f'{"="*100}\n\n\n')
    
