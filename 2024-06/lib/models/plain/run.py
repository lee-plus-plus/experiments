import argparse
import torch

from ...backbone.mlp import MlpNet
from ...utils import init_cuda_environment
from ...table_dataset import build_dataset, supported_datasets
from .main import train_model


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='PLAIN')
    # Optimization options

    parser.add_argument('--dataset', default='music_emotion', type=str,
                        choices=supported_datasets())
    parser.add_argument('--noise-rate', type=float, default=0.2,
                        help='corruption rate, should be less than 1')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--hidden-layer-size', type=str, default='64,64')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='train batchsize')

    parser.add_argument('--num-neighbors', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.7)

    # main parameters
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--eta', type=float, default=1)

    parser.add_argument('--gpu', default='0', type=str,
                        help='gpu device be used')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed for initialization')

    args = parser.parse_args()

    args.hidden_layer_size = [
        int(n) for n in args.hidden_layer_size.split(',')]

    return args


def run(args=None):
    args = parse_args(args)
    print(__file__)
    print(args)

    init_cuda_environment(seed=args.seed, device=args.gpu)

    print('creating dataset...')
    train_dataset, valid_dataset = build_dataset(
        name=args.dataset,
        add_partial_noise=True, noise_rate=args.noise_rate,
        add_index=True, scale=True, getitems=True,
    )

    print("len(train_dataset)): ", len(train_dataset))
    print("len(valid_dataset)): ", len(valid_dataset))

    # Pytorch Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        collate_fn=lambda x: x
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=False,
        collate_fn=lambda x: x
    )

    # Setup model
    print('creating model...')
    model = MlpNet(
        in_features=train_dataset.dim_features,
        out_features=train_dataset.num_classes,
        hidden_layer_sizes=args.hidden_layer_size
    ).cuda()

    # Actuall Training
    print('begin training')
    train_model(
        model, train_loader, valid_loader,
        lr=args.lr, epochs=args.epochs,
        alpha=args.alpha, beta=args.beta, eta=args.eta,
        num_neighbors=args.num_neighbors, threshold=args.threshold
    )

    print('train done.')


if __name__ == '__main__':
    run(args=None)
