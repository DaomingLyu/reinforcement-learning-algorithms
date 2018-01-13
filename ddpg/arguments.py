import argparse
import torch

# define some parameters which will be used....
def achieve_args():
	parse = argparse.ArgumentParser()
	parse.add_argument('--env', default='Pendulum-v0', help='the name of the training environemnt')
	parse.add_argument('--policy_lr', type=float, default=0.0001, help='the learning rate')
	parse.add_argument('--value_lr', type=float, default=0.001, help='the learning rate')
	parse.add_argument('--tau', type=float, default=0.001, help='used for soft-update')
	parse.add_argument('--gamma', type=float, default=0.99, help='discount factor')
	parse.add_argument('--seed', type=int, default=1, help='the random seed of the torch')
	parse.add_argument('--buffer_size', type=int, default=1000000, help='the size of the buffer')
	parse.add_argument('--max_time_step', type=int, default=1000, help='the max_time_step per eposide')
	parse.add_argument('--noise_scale', type=float, default=0.3, help='the scale of the noise')
	parse.add_argument('--final_noise_scale', type=float, default=0.3, help='the final noise scale')
	parse.add_argument('--observate_time', type=int, default=1000, help='the observate_time')
	parse.add_argument('--batch_size', type=int, default=64, help='the batch_size')
	parse.add_argument('--save_dir', default='saved_models/', help='the folder which save the models')
	parse.add_argument('--soft_update_step', type=int, default=5, help='the step to update the target network')
	parse.add_argument('--cuda', type=int, default=1, help='if use the cuda')


	args = parse.parse_args()

	args.cuda = args.cuda * int(torch.cuda.is_available())
	args.save_dir = args.save_dir + args.env + '/'

	return args



