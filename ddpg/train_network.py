import gym
import mujoco_py
from ddpg_agent import ddpg_brain
import arguments

# achieve the arguments...
args = arguments.achieve_args()

# start to set parameters and train the network...
env = gym.make(args.env)

ddpg_man = ddpg_brain(env=env, 
						policy_lr=args.policy_lr,
						value_lr=args.value_lr,
						tau=args.tau,
						gamma=args.gamma,
						buffer_size=args.buffer_size,
						max_time_step=args.max_time_step,
						observate_time=args.observate_time,
						batch_size=args.batch_size,
						path=args.save_dir,
						soft_update_step=args.soft_update_step,
						use_cuda=args.cuda
						)


ddpg_man.train_network()

