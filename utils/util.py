import random
import torch
import numpy as np
from tqdm import tqdm
from torch_sparse import SparseTensor
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from silearn.graph import GraphSparse
from silearn.optimizer.enc.partitioning.propagation import OperatorPropagation
from silearn.model.encoding_tree import Partitioning
from sklearn import metrics
from munkres import Munkres
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
from torch_scatter import scatter_mean
from torch_geometric.utils import subgraph

matplotlib.use('Agg')
import matplotlib.pyplot as plt

#try:
#    from sklearnex import patch_sklearn
#except:
def patch_sklearn(): return

def get_graph_stats(node_ids, edge_index):
    """
    计算给定节点集在原图中的诱导子图特征
    :param node_ids: Tensor, 当前batch包含的节点ID集合 (n_id 或 batch_ex)
    :param edge_index: Tensor, 全局图的边索引 [2, E]
    :return: dict, 包含节点数、内部边数和平均度数
    """
    # 确保节点ID是唯一的（以防采样器返回的有重复）
    unique_nodes = torch.unique(node_ids)
    num_nodes = unique_nodes.numel()
    
    if num_nodes == 0:
        return {"nodes": 0, "edges": 0, "avg_degree": 0.0}
    
    # 提取由这些节点组成的诱导子图（即边两端的节点都在 unique_nodes 中）
    # relabel_nodes=False 保持原始节点索引，计算更快
    sub_edge_index, _ = subgraph(unique_nodes, edge_index, relabel_nodes=False)
    
    # 内部边数
    num_edges = sub_edge_index.size(1)
    
    # 平均度数 
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0.0
    
    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "avg_degree": avg_degree
    }

class Cluster_Metrics:
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)

        if numclass1 != numclass2:
            print(numclass1, numclass2, 'Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self, tqdm):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        fms = metrics.fowlkes_mallows_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        return acc, nmi, adjscore, fms, f1_macro, f1_micro

    @staticmethod
    def plot(X, fig, col, size, true_labels):
        ax = fig.add_subplot(1, 1, 1)
        for i, point in enumerate(X):
            ax.scatter(point[0], point[1], lw=0, s=size, c=col[true_labels[i]])

    def plotClusters(self, tqdm, hidden_emb, true_labels):
        tqdm.write('Start plotting using TSNE...')
        # Doing dimensionality reduction for plotting
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(hidden_emb)
        # Plot figure
        fig = plt.figure()
        self.plot(X_tsne, fig, ['red', 'green', 'blue', 'brown', 'purple', 'yellow', 'pink', 'orange'], 40, true_labels)
        plt.axis("off")
        fig.savefig("plot.png", dpi=120)
        tqdm.write("Finished plotting")


def get_sim(batch, adj, wt=20, wl=3):
    rowptr, col, _ = adj.csr()
    batch_size = batch.shape[0]
    batch_repeat = batch.repeat(wt)
    rw = adj.random_walk(batch_repeat, wl)[:, 1:]

    if not isinstance(rw, torch.Tensor):
        rw = rw[0]
    rw = rw.t().reshape(-1, batch_size).t()

    row, col, val = [], [], []
    for i in range(batch.shape[0]):
        rw_nodes, rw_times = torch.unique(rw[i], return_counts=True)
        row += [batch[i].item()] * rw_nodes.shape[0]
        col += rw_nodes.tolist()
        val += rw_times.tolist()

    unique_nodes = list(set(row + col))
    subg2g = dict(zip(unique_nodes, list(range(len(unique_nodes)))))

    row = [subg2g[x] for x in row]
    col = [subg2g[x] for x in col]
    idx = torch.tensor([subg2g[x] for x in batch.tolist()])

    adj_ = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), value=torch.tensor(val),
                        sparse_sizes=(len(unique_nodes), len(unique_nodes)))

    adj_batch, _ = adj_.saint_subgraph(idx)
    adj_batch = adj_batch.set_diag(0.)
    # src, dst = dict_r[idx[adj_batch.storage.row()[3].item()].item()], dict_r[idx[adj_batch.storage.col()[3].item()].item()]
    return batch, adj_batch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clustering(feature, n_clusters, true_labels, kmeans_device='cpu', batch_size=100000, tol=1e-4, device=torch.device('cpu'), spectral_clustering=False):
    if spectral_clustering:
        if isinstance(feature, torch.Tensor):
            feature = feature.numpy()
        print("spectral clustering on cpu...")
        patch_sklearn()
        Cluster = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', random_state=0)
        f_adj = np.matmul(feature, np.transpose(feature))
        predict_labels = Cluster.fit_predict(f_adj)
    else:
        if kmeans_device == 'cuda':
            from utils.batch_kmeans_cuda import kmeans
            if isinstance(feature, np.ndarray):
                feature = torch.tensor(feature)
            print("kmeans on gpu...")
            predict_labels, _ = kmeans(
                X=feature, num_clusters=n_clusters, batch_size=batch_size, tol=tol, device=device)
            predict_labels = predict_labels.numpy()
        else:
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
            print("kmeans on cpu...")
            patch_sklearn()
            Cluster = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=20)
            predict_labels = Cluster.fit_predict(feature)

    cm = Cluster_Metrics(true_labels, predict_labels)
    acc, nmi, adjscore, fms, f1_macro, f1_micro = cm.evaluationClusterModelFromLabel(
        tqdm)
    return acc, nmi, adjscore, f1_macro, f1_micro


def get_mask(adj):
    batch_mean = adj.mean(dim=1)
    mean = batch_mean[torch.LongTensor(adj.storage.row())]
    mask = (adj.storage.value() - mean) > - 1e-10
    row, col, val = adj.storage.row()[mask], adj.storage.col()[
        mask], adj.storage.value()[mask]
    adj_ = SparseTensor(row=row, col=col, value=val,
                        sparse_sizes=(adj.size(0), adj.size(1)))
    return adj_

def get_mask1(adj):
    # 1. 直接从 SparseTensor 获取数据，避免直接对 adj 对象做 dim=1 的操作
    row = adj.storage.row()
    value = adj.storage.value()
    
    # 2. 使用 scatter_mean 计算每一行的均值 (对应原先的 dim=1)
    # 这不会创建稠密矩阵，只会根据边信息进行聚合
    # adj.size(0) 是行数
    batch_mean = scatter_mean(value, row, dim=0, dim_size=adj.size(0))
    
    # 3. 映射均值到每一条边上
    mean_per_edge = batch_mean[row]
    
    # 4. 生成掩码并过滤
    mask = (value - mean_per_edge) > -1e-10
    
    # 5. 构造新的 SparseTensor
    # 这里的索引操作都在现有的存储上完成，开销极小
    new_row = row[mask]
    new_col = adj.storage.col()[mask]
    new_val = value[mask]
    
    adj_ = SparseTensor(
        row=new_row, 
        col=new_col, 
        value=new_val,
        sparse_sizes=(adj.size(0), adj.size(1))
    )
    return adj_

def get_se_mask(adj, device):
    device = torch.device(device)
    row, col, val = adj.storage.row().to(device), adj.storage.col().to(device), adj.storage.value().to(device).float()
    val = torch.ones_like(adj.storage.value(), device=device, dtype=torch.float)
    edge_index = torch.stack([row, col], dim=1).to(device)
    num_nodes = adj.size(0)
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, row, val)
    dist = deg*2  # dist/2=di
    dist = dist.to(device)
    dist = dist / (2 * val.sum())

    graph = GraphSparse(edge_index, val, dist)
    optim = OperatorPropagation(Partitioning(graph, None))
    optim.perform(p=0.01)
    division = optim.enc.node_id
    totol_comm = torch.max(division) + 1

    mask = (division[row] == division[col])
    row, col, val = row[mask], col[mask], val[mask]
    adj_ = SparseTensor(row=row, col=col, value=val,
                        sparse_sizes=(adj.size(0), adj.size(1)))
    return adj_, totol_comm, (len(edge_index), len(mask))


def scale(z: torch.Tensor):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / ((zmax - zmin) + 1e-20)
    z_scaled = z_std
    return z_scaled

def z_score_scale(z: torch.Tensor, dim=0):
    mean = z.mean(dim=dim, keepdim=True)
    std = z.std(dim=dim, keepdim=True)
    return (z - mean) / (std + 1e-8)

def cal_se_hard(deg, comm, edge_index):
    comm_ids = torch.unique(comm)
    k = len(comm_ids)
    src,dst = edge_index[0], edge_index[1]
    vol_G = deg.sum(0)
    vol_c = torch.zeros(k, device=deg.device)
    vol_in = torch.zeros(k, device=deg.device)
    
    for idx, c in enumerate(comm_ids):
        mask = (comm == c)
        nodes = mask.nonzero(as_tuple=True)[0]
        
        vol_c[idx] = deg[nodes].sum()
        
        # 社区内部边
        edge_mask = mask[src] & mask[dst]
        vol_in[idx] = edge_mask.sum()
    
    g_c = vol_c - vol_in
    SE_c = -(g_c / vol_G) * torch.log2(vol_c / vol_G)
    SE_i = torch.zeros_like(deg, dtype=torch.float32)
    for idx, c in enumerate(comm_ids):
        nodes = (comm == c).nonzero(as_tuple=True)[0]
        SE_i[nodes] = -(deg[nodes] / vol_G) * torch.log2(deg[nodes] / vol_c[idx])

    return SE_c, SE_c.sum() + SE_i.sum()