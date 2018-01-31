import gym
from dqn_agent import dqn_brain
import arguments


# achieve the arguments...
args = arguments.achieve_args()

# start to create the environment...
env = gym.make(args.env)

dqn_man = dqn_brain(env = env, 
                    lr = args.lr,
                    gamma = args.gamma,
                    buffer_size = args.buffer_size,
                    init_exploration = args.init_exploration,
                    final_exploration = args.final_exploration, 
                    exploration_step = args.exploration_steps,
                    batch_size = args.batch_size, 
                    save_dir = args.save_dir,
                    hard_update_step = args.hard_update_step, 
                    observate_time = args.observate_time, 
                    use_cuda = args.cuda)


dqn_man.train_network()
