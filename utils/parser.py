import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="UnLC")

    # dataset para
    parser.add_argument('--dataset', type=str, default='Reddit')
    parser.add_argument('--noise_rate', type=float, default='0', help='noise')

    # sample para    
    parser.add_argument('--commNum', type=int, default=1200, help='num of comms')
    parser.add_argument('--p', type=float, default=0.09, help='Parallelism rate of SE')
    parser.add_argument('--theta', type=int, default=20, help='threshold of comm size')
    parser.add_argument('--batchsize', type=int, default=2048, help='init batchsize')
    parser.add_argument('--wt', type=int, default=20)
    parser.add_argument('--wl', type=int, default=5)

    # learning para
    parser.add_argument('--dropout', type=float, default=0, help='')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0, help='weight_decay')
    parser.add_argument('--epochs', type=int, default=201)

    # model para
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--hidden_channels', type=str, default='1024,256')
    parser.add_argument('--size', type=str, default='10,10', help='')
    parser.add_argument('--tau', type=float, default=0.5, help='temperature in loss')
    parser.add_argument('--ns', type=float, default=0.5, help='parameter of leaky_relu')
    parser.add_argument('--projection', type=str, default='')
    parser.add_argument('--base_model', type=str, default='SAGEConv',help='SAGEConv,GCNConv')

    # kmeans para
    parser.add_argument('--kmeans_device', type=str,
                        default='cpu', help='kmeans device, cuda or cpu')
    parser.add_argument('--kmeans_batch', type=int, default=-1,
                        help='batch size of kmeans on GPU, -1 means full batch')


    return parser.parse_args()