import argparse
import os
from solver import Solver

def main(args):
    solver = Solver(root = args.root,
                    result_dir = args.result_dir,
                    weight_dir = args.weight_dir,
                    load_weight = args.load_weight,
                    batch_size = args.batch_size,
                    test_size = args.test_size,
                    img_size = args.img_size,
                    num_epoch = args.num_epoch,
                    save_every = args.save_every,
                    save_epoch = args.save_epoch,
                    lr = args.lr,
                    beta_1 = args.beta_1,
                    beta_2 = args.beta_2,
                    lambda_recon = args.lambda_recon,
                    logdir = args.logdir,
                    lr_decay_iters = args.lr_decay_iters,
                    pretrained_model = args.pretrained_model
                    )
                    
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='C:/Dataset/NCDataset', 
                        help='Data location')
    parser.add_argument('--result_dir', type=str, default='test', 
                        help='Result images location')
    parser.add_argument('--weight_dir', type=str, default='weights', 
                        help='Weight location')
    parser.add_argument('--logdir', type=str, default='logs', 
                        help='logger location')
    parser.add_argument('--batch_size', type=int, default=25, 
                        help='Training batch size')
    parser.add_argument('--test_size', type=int, default=1, 
                        help='Test batch size')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='Image size')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9, 
                        help='Beta1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.99, 
                        help='Beta2 for Adam')
    parser.add_argument('--lambda_recon', type=float, default=1e-2, 
                        help='Lambda for reconstruction loss')
    parser.add_argument('--num_epoch', type=int, default=400, 
                        help='Number of epoch')
    parser.add_argument('--save_every', type=int, default=23,  # 150
                        help='How often do you want to see the result?')
    parser.add_argument('--save_epoch', type=int, default=5, 
                        help='How often do you want to save the model?')
    parser.add_argument('--load_weight', type=bool, default=True,
                        help='Load weight or not')
    parser.add_argument('--lr_decay_iters', type=int, default=30, 
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--pretrained_model', type=str, default='C:/Code/black2color/weights/checkpoint_5_epoch.pkl', 
                        help='load pretrained model')
    args = parser.parse_args(args=[])
    main(args)