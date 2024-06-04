from test1223 import *
import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='Multi-label Image Classification Model (2023-12-23)')
    
    parser.add_argument('--backbone', default='resnet18', type=str, choices=['resnet18', 'resnet101'])
    parser.add_argument('--dataset', default='coco2014', type=str, choices=['coco2014', 'coco2017', 'voc2007', 'voc2012', 'nus_wide'])
    parser.add_argument('--pretrained', default=False, type=bool)
    parser.add_argument('--image-size', default=224, type=int, choices=[224, 448])

    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--batch-size', default='128', type=int)
    parser.add_argument('--num-epoches', default='50', type=int)
    parser.add_argument('--show-progress', default=False, type=bool)

    parser.add_argument('--train-policy', default='vanilla', choices=['vanilla', 'mulcon'])
    parser.add_argument('--weight-contrastive', default=10, type=float)


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parser_args()
    print(f'{args = }')

    # ### Initalize

    init_cuda_environment(seed=123, device=args.gpu)

    # dataset
    train_dataset = build_dataset(args.dataset, 'train', image_size=args.image_size)
    test_dataset = build_dataset(args.dataset, 'test', image_size=args.image_size)

    from torch.utils.data import DataLoader
    batch_size = args.batch_size

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=8, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=8, drop_last=True
    )

    # models
    num_classes = train_dataset.num_classes
    dim_encoding = 512
    dim_contrastive = 512

    model = MyNet(
        num_classes=num_classes, 
        dim_encoding=dim_encoding, 
        dim_contrastive=dim_contrastive, 
        backbone=args.backbone,
        pretrained=args.pretrained
    ).cuda()

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=1e-2,
        momentum=0.9,
        weight_decay=1e-4
    )

    idx_queue = 0
    size_queue = 4096
    embed_queue = torch.zeros((size_queue, num_classes, dim_contrastive)).cuda()
    y_true_queue = torch.zeros((size_queue, num_classes)).cuda()


    ## Begin Training

    import warnings
    warnings.filterwarnings("ignore")

    meter_train = LastMeter()
    meter_valid = LastMeter()
    epoch = 0

    if args.train_policy == 'vanilla':
        
        # ### Vanilla Training
        print('start Vanilla Training:')

        for e in range(50):
            meter_epoch_train = train_epoch_vanilla(
                train_loader, optimizer, model, 
                embed_queue, y_true_queue, idx_queue, size_queue, 
                show_progress=args.show_progress
            )
            meter_epoch_valid = eval_epoch_vanilla(
                test_loader, optimizer, model, 
                show_progress=args.show_progress
            )
            
            print(f'epoch {epoch:>2d} (train): {meter_epoch_train}')
            print(f'epoch {epoch:>2d} (valid): {meter_epoch_valid}')
            
            meter_train.update(epoch=epoch, **meter_epoch_train.average_value())
            meter_valid.update(epoch=epoch, **meter_epoch_valid.average_value())
            
            epoch += 1

    elif args.train_policy == 'mulcon':

        # ### MulCon Training
        print('start MulCon Training:')

        for e in range(50):
            meter_epoch_train = train_epoch_mulcon(
                train_loader, optimizer, model, 
                embed_queue, y_true_queue, idx_queue, size_queue,
                show_progress=args.show_progress,
                weight_contrastive=args.weight_contrastive
            )
            meter_epoch_valid = eval_epoch_mulcon(
                test_loader, optimizer, model,
                embed_queue, y_true_queue, idx_queue, size_queue, 
                show_progress=args.show_progress,
                weight_contrastive=args.weight_contrastive
            )
            
            print(f'epoch {epoch:>2d} (train): {meter_epoch_train}')
            print(f'epoch {epoch:>2d} (valid): {meter_epoch_valid}')
            
            meter_train.update(epoch=epoch, **meter_epoch_train.average_value())
            meter_valid.update(epoch=epoch, **meter_epoch_valid.average_value())
            
            epoch += 1


