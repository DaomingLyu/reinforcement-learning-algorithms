import argparse
import torch

# define some parameters which will be used....
def achieve_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--env', default='FlappyBird-v0', help='the name of the training environemnt')
    parse.add_argument('--lr', type=float, default=1e-5)
    parse.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parse.add_argument('--seed', type=int, default=1, help='the random seed of the torch')
    parse.add_argument('--buffer_size', type=int, default=50000, help='the size of the buffer')
    parse.add_argument('--init_exploration', type=float, default=0.1, help='the inital epsilon')
    parse.add_argument('--final_exploration', type=float, default=0.02, help='the final epsilon')
    parse.add_argument('--exploration_steps', type=int, default=1000000, help='the exploration frames')
    parse.add_argument('--observate_time', type=int, default=1000, help='the observate_time')
    parse.add_argument('--batch_size', type=int, default=32, help='the batch_size')
    parse.add_argument('--save_dir', default='saved_models/', help='the folder which save the models')
    parse.add_argument('--hard_update_step', type=int, default=500, help='the step to update the target network')
    parse.add_argument('--cuda', type=int, default=1, help='if use the cuda')

    args = parse.parse_args()

    args.cuda = args.cuda * int(torch.cuda.is_available())
    args.save_dir = args.save_dir + args.env + '/'

    return args



