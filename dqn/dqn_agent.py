import numpy as np 
import models
import torch
from torch.autograd import Variable
import cv2
import random
import gym_ple

"""
This is the implementation of DQN

2017-01-17

author: Tianhong Dai

"""

class dqn_brain:
    def __init__(self, args, env):
        # get the arguments...
        self.args = args
        # init the parameters....
        self.env = env    
        # check if use the cuda to train the network...
        self.use_cuda = torch.cuda.is_available() and self.args.cuda 
        print('The cuda is avaiable: ' + str(torch.cuda.is_available()))
        print('If use the cuda: ' + str(self.args.cuda))

        # get the number of actions....
        # action space fot the FlappyBird....
        self.action_space = [0, 1]
        self.num_actions = len(self.action_space)

        # build up the network.... and the target network
        self.deep_q_network = models.Deep_Q_Network(self.num_actions)
        self.target_network = models.Deep_Q_Network(self.num_actions)
        # decide if put into the cuda...

        if self.use_cuda:
            self.deep_q_network.cuda()
            self.target_network.cuda()

        # init the parameters of the target network...
        self.target_network.load_state_dict(self.deep_q_network.state_dict())

        # init the optimizer
        self.optimizer = torch.optim.Adam(self.deep_q_network.parameters(), lr=self.args.lr)

    # this is used to train the network...
    def train_network(self):
        # init the memory buff...
        brain_memory = []
        num_of_episode = 0
        global_step = 0
        update_step_counter = 0
        reward_mean = None
        epsilon = self.args.init_exploration
        loss = 0

        while True:
            state = self.env.reset()
            state = self._pre_processing(state)
            # for the first state we need to stack them together....
            state = np.stack((state, state, state, state), axis=0)
            # clear the rewrad_sum...
            reward_sum = 0
            pipe_num = 0
            # I haven't set a max step here, but you could set it...
            while True:
                state_tensor = torch.Tensor(state).unsqueeze(0)
                if self.use_cuda:
                    state_tensor = Variable(state_tensor).cuda()
                else:
                    state_tensor = Variable(state_tensor)

                _, _, actions = self.deep_q_network(state_tensor)

                action_selected = self._selected_the_actions(actions, epsilon)
                # input the action into the environment...
                state_, reward, done, _ = self.env.step(self.action_space[action_selected])

                # process the output state...
                state_ = self._pre_processing(state_)

                # concatenate them together...
                state_temp = state[0:3, :, :].copy()
                state_ = np.expand_dims(state_, 0)
                state_ = np.concatenate((state_, state_temp), axis=0)

                # wrapper the reward....
                reward = self._reward_wrapper(reward)

                # add the pip num...
                if reward > 0:
                    pipe_num += 1

                reward_sum += reward
                global_step += 1

                # store the transition...
                brain_memory.append((state, state_, reward, done, action_selected))

                if len(brain_memory) > self.args.buffer_size:
                    brain_memory.pop(0)

                if global_step >= self.args.observate_time:
                    mini_batch = random.sample(brain_memory, self.args.batch_size)
                    loss = self._update_network(mini_batch)
                    update_step_counter += 1
                    # up date the target network...
                    if update_step_counter % self.args.hard_update_step == 0:
                        self._hard_update_target_network(self.deep_q_network, self.target_network)

                # process the epsilon
                if global_step <= self.args.exploration_steps:
                    epsilon -= (self.args.init_exploration - self.args.final_exploration) / self.args.exploration_steps

                if done:
                    break

                state = state_

            # expoential weighted average...
            reward_mean = pipe_num if reward_mean is None else reward_mean * 0.99 + pipe_num * 0.01

            if num_of_episode % self.args.display_interval == 0:
                print('The episode number is ' + str(num_of_episode) + ', the reward mean is ' + str(reward_mean) + \
                                                                                            ', and the loss ' + str(loss))

            if num_of_episode % self.args.save_interval == 0:
                save_path = self.args.save_dir + 'model_' + str(num_of_episode) + '.pt'
                torch.save(self.deep_q_network.state_dict(), save_path)

            num_of_episode += 1

    # process the image..
    def _pre_processing(self, x):
        x = x[:, :, (2, 1, 0)]
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = np.float32(x) / 255
        x = cv2.resize(x, (84, 84))
        
        return x

    def _reward_wrapper(self, reward):
        if reward < 0:
            reward = -1
        elif reward > 0:
            reward = 1

        return reward

    def _selected_the_actions(self, action, epsilon):
        # transfer the action from the gpu to cpu
        action_selected = action.data.cpu().numpy()[0]
        action_selected = int(action_selected)

        # greedy...
        dice = random.uniform(0, 1)

        if dice >= 1 - epsilon:
            action_selected = random.randint(0, self.num_actions - 1)

        return action_selected

    # this is used to update the q_learning_network...
    def _update_network(self, mini_batch):
        # process the data...
        state_batch = np.array([element[0] for element in mini_batch])
        state_batch_tensor = torch.Tensor(state_batch)

        state_next_batch = np.array([element[1] for element in mini_batch])
        state_next_batch_tensor = torch.Tensor(state_next_batch)

        reward_batch = np.array([element[2] for element in mini_batch])
        reward_batch_tensor = torch.Tensor(reward_batch).unsqueeze(1)

        done_batch = np.array([float(element[3]) for element in mini_batch])
        done_batch = 1 - done_batch
        done_batch_tensor = torch.Tensor(done_batch).unsqueeze(1)

        action_batch = np.array([element[4] for element in mini_batch])
        action_batch_tensor = torch.LongTensor(action_batch).unsqueeze(1)

        # put the tensor into the gpu...

        if self.use_cuda:
            state_batch_tensor = Variable(state_batch_tensor).cuda()
            state_next_batch_tensor = Variable(state_next_batch_tensor).cuda()
            reward_batch_tensor = Variable(reward_batch_tensor).cuda()
            done_batch_tensor = Variable(done_batch_tensor).cuda()
            action_batch_tensor = Variable(action_batch_tensor).cuda()
        else:   
            state_batch_tensor = Variable(state_batch_tensor)
            state_next_batch_tensor = Variable(state_next_batch_tensor)
            reward_batch_tensor = Variable(reward_batch_tensor)
            done_batch_tensor = Variable(done_batch_tensor)
            action_batch_tensor = Variable(action_batch_tensor)


        # calculate the target value....
        _, q_max_value, _ = self.target_network(state_next_batch_tensor)
        q_max_value = q_max_value.unsqueeze(1)

        target = reward_batch_tensor + self.args.gamma * q_max_value * done_batch_tensor
        # remove the target from the computation graph...
        target = target.detach()
        # calculate the loss
        Q_value, _, _ = self.deep_q_network(state_batch_tensor)

        real_Q_value = Q_value.gather(1, action_batch_tensor)

        loss = (target - real_Q_value).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.data.cpu().numpy()[0]

    # hard update the target network....
    def _hard_update_target_network(self, source, target):
        for param, param_target in zip(source.parameters(), target.parameters()):
            param_target.data.copy_(param.data)

# -------------------------- Here is to test the network.... ------------------------------#

    def test_network(self):
        model_path = self.args.save_dir + 'model.pt'
        self.deep_q_network.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.deep_q_network.eval()

        while True: 
            state = self.env.reset()
            state = self._pre_processing(state)
            # for the first state we need to stack them together....
            state = np.stack((state, state, state, state), axis=0)
            # clear the rewrad_sum...
            pipe_sum = 0
            # I haven't set a max step here, but you could set it...
            while True:
                self.env.render()
                state_tensor = torch.Tensor(state).unsqueeze(0)
                if self.use_cuda:
                    state_tensor = Variable(state_tensor).cuda()
                else:
                    state_tensor = Variable(state_tensor)

                _, _, actions = self.deep_q_network(state_tensor)

                # action...deterministic...
                action_selected = int(actions.data.numpy()[0])

                state_, reward, done, _ = self.env.step(self.action_space[action_selected])
                if reward > 0:
                    pipe_sum += 1

                # process the output state...
                state_ = self._pre_processing(state_)

                # concatenate them together...
                state_temp = state[0:3, :, :].copy()
                state_ = np.expand_dims(state_, 0)
                state_ = np.concatenate((state_, state_temp), axis=0)

                if done:
                    break
                    
                state = state_

            print('In this episode, the bird totally pass ' + str(pipe_sum) + ' pipes!')

