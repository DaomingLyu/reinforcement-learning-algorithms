import numpy as np 
import torch
import models
from torch.autograd import Variable 
import models
import pyro
import pyro.distributions as dist
from running_state import ZFilter

# author: Tianhong Dai
# The implementation of Proximal Policy Optimization...
# This is the simplest PPO algorithm...
# The beta distribution will also be used in this code....

class ppo_brain():
    def __init__(self, env, policy_network, value_network, gamma=0.99, policy_lr=0.0005, value_lr=0.0005, tau=0.95, 
                    value_update_step=10, policy_update_step=10, 
                    epsilon=0.2, batch_size=64, use_cuda=True, saved_path='saved_models/Walker2d-v1/'):
        # define the parameters...
        self.env = env
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.tau = tau
        self.value_update_step = value_update_step
        self.policy_update_step = policy_update_step
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.saved_path = saved_path
        # check if cuda is avaiable...
        self.use_cuda = torch.cuda.is_available() and use_cuda

        print('The cuda is avaiable: ' + str(self.use_cuda))

        # define the network...
        self.policy_network = policy_network
        self.value_network = value_network

        if self.use_cuda:
            self.policy_network.cuda()
            self.value_network.cuda()

        # define the optimizer
        self.optimizer_value = torch.optim.Adam(self.value_network.parameters(), lr=self.value_lr)
        self.optimizer_policy = torch.optim.Adam(self.policy_network.parameters(), lr=self.policy_lr)

        # init the Filter...
        self.running_state = ZFilter((self.env.observation_space.shape[0],), clip=5)

    # start to train the network...
    def train_network(self):
        num_of_episode = 0
        while True:
            # reset up a memory
            brain_memory = []
            batch_reward = 0
            # every 100 eposide update once ...
            for ep in range(self.batch_size):
                # reset the state....
                state = self.env.reset()
                state = self.running_state(state)
                reward_sum = 0
                # time steps in one game...
                while True:
                    state_tensor = torch.Tensor(state).view(1, -1)
                    if self.use_cuda:
                        state_tensor = Variable(state_tensor).cuda()
                    else:
                        state_tensor = Variable(state_tensor)

                    # get the parameters of the action's gaussian distribution....
                    action_alpha, action_beta = self.policy_network(state_tensor)
                    # convert the variable from cuda to cpu, because cuda can not convert to numpy...
                    action_selected = self.action_selection(action_alpha, action_beta)
                    action_selected_cpu = action_selected.data.cpu().numpy()[0]
                    # input the action and get the feedback...
                    action_actual = action_selected_cpu.copy()
                    # in the range [-1, 1]
                    action_actual = -1 + action_actual * 2
                    state_, reward, done, _  = self.env.step(action_actual)
                    # sum of the reward
                    reward_sum += reward

                    # process the state_
                    state_ = self.running_state(state_)
                    # store the transitions...
                    brain_memory.append((state, action_selected_cpu, reward, done))

                    if done:
                        break
                    state = state_
                # sum the reward for each batch....
                batch_reward += reward_sum
            # now we can do the update...
            value_loss, policy_loss = self.update_the_network(brain_memory)
            batch_reward = batch_reward / self.batch_size

            if num_of_episode % 1 == 0:
                print('The iteration number is ' + str(num_of_episode) + ' and the reward mean is ' + str(batch_reward) + 
                      ', the value_loss is ' + str(value_loss) + ', the policy_loss is ' + str(policy_loss))

            if num_of_episode % 100 == 0:
                path_gpu = self.saved_path + 'policy_net_model_'
                path_gpu = path_gpu + str(num_of_episode) + '.pt'
                torch.save([self.policy_network.state_dict(), self.running_state], path_gpu)
                print('-----------------------------------------------------')
                print('models has been saved!!!')
                print('-----------------------------------------------------')
            num_of_episode += 1

    def update_the_network(self, brain_memory):
        # process the state batch...
        state_batch = np.array([element[0] for element in brain_memory])
        state_batch_tensor = torch.Tensor(state_batch)

        # process the reward batch...
        reward_batch = np.array([element[2] for element in brain_memory])
        reward_batch_tensor = torch.Tensor(reward_batch)
        
        # process the done batch...
        done_batch = [element[3] for element in brain_memory]

        # process the action batch
        action_batch = np.array([element[1] for element in brain_memory])
        action_batch_tensor = torch.Tensor(action_batch)

        # put them onto the gpu or cpu...
        if self.use_cuda:
            state_batch_tensor = Variable(state_batch_tensor).cuda()
            action_batch_tensor = Variable(action_batch_tensor).cuda()

        else:
            state_batch_tensor = Variable(state_batch_tensor)
            action_batch_tensor = Variable(action_batch_tensor)

        # calculate the estimated return value of the state...
        returns, advantages = self.calculate_the_discounted_reward(reward_batch_tensor, done_batch, state_batch_tensor)

        # update the value network!!!
        loss_value = self.update_the_value_network(returns, state_batch_tensor)
        loss_policy = self.update_the_policy_network(state_batch_tensor, advantages, action_batch_tensor)

        return loss_value.data.cpu().numpy()[0], loss_policy.data.cpu().numpy()[0]

    # selection the actions according to the gaussian distribution...
    def action_selection(self, action_alpha, action_beta, exploration=True):
        if exploration:
            action = dist.beta(action_alpha, action_beta)
        else:
            action = dist.Beta(action_alpha, action_beta).analytic_mean()
        #print action
        return action

    # calculate the discounted reward...
    def calculate_the_discounted_reward(self, reward_batch_tensor, done_batch, state_batch_tensor):
        predicted_value = self.value_network(state_batch_tensor)
        # detach from the graph...
        predicted_value = predicted_value.detach()

        returns = torch.Tensor(len(done_batch), 1)
        advantages = torch.Tensor(len(done_batch), 1)

        previous_return = 0
        previous_value = 0

        for index in reversed(range(len(done_batch))):
            if done_batch[index]:
                returns[index, 0] = reward_batch_tensor[index]
                advantages[index, 0] = returns[index, 0] - predicted_value.data[index, 0]

            else:
                returns[index, 0] = reward_batch_tensor[index] + self.gamma * previous_return
                advantages[index, 0] = returns[index, 0] - predicted_value.data[index, 0]

            previous_return = returns[index, 0]

        # normalize the advantages...
        advantages = (advantages - advantages.mean()) / advantages.std()    

        return returns, advantages

    # update the value network... or critic network...
    def update_the_value_network(self, returns, state_batch_tensor):        
        if self.use_cuda:
            targets = Variable(returns).cuda()
        else:
            targets = Variable(returns)

        for _ in range(self.value_update_step):
            predicted_value = self.value_network(state_batch_tensor)
            loss_value = (predicted_value - targets).pow(2).mean()

            self.optimizer_value.zero_grad()
            loss_value.backward()
            # update
            self.optimizer_value.step()

        return loss_value

    # update the actor network....
    def update_the_policy_network(self, state_batch_tensor, advantages, action_batch_tensor):

        action_alpha_old, action_beta_old = self.policy_network(state_batch_tensor)
        old_beta_dist = dist.Beta(action_alpha_old, action_beta_old)
        old_action_prob = old_beta_dist.batch_log_pdf(action_batch_tensor)
        old_action_prob = old_action_prob.detach()

        if self.use_cuda:
            advantages = Variable(advantages).cuda()
        else:
            advantages = Variable(advantages)

        for _ in range(self.policy_update_step):
            action_alpha, action_beta = self.policy_network(state_batch_tensor)
            new_beta_dist = dist.Beta(action_alpha, action_beta)
            new_action_prob = new_beta_dist.batch_log_pdf(action_batch_tensor)
            # calculate the ratio
            ratio = torch.exp(new_action_prob - old_action_prob)

            # calculate the surr
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

            loss_policy = -torch.min(surr1, surr2).mean()

            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            # update
            self.optimizer_policy.step()

        return loss_policy

# ------------------------------------------------------------

    # here is used to test the network....
    def test_network(self, model_path):

        # load the models
        policy_model, fiter_model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.policy_network.load_state_dict(policy_model)
        self.policy_network.eval()

        while True:
            state = self.env.reset()
            state = self.test_filter(state, fiter_model.rs.mean, fiter_model.rs.std)
            reward_sum = 0

            for _ in range(10000):
                self.env.render()
                state_tensor = torch.Tensor(state).view(1, -1)
                state_tensor = Variable(state_tensor)
                # get the actions...
                action_alpha, action_beta = self.policy_network(state_tensor)
                action_selected = self.action_selection(action_alpha, action_beta, exploration=False)
                action_selected_numpy = action_selected.data.numpy()[0]
                action_selected_numpy = -1 + action_selected_numpy * 2

                # input the action into the env...
                state_, reward, done, _ = self.env.step(action_selected_numpy)

                reward_sum += reward

                state_ = self.test_filter(state_, fiter_model.rs.mean, fiter_model.rs.std)

                if done:
                    break

                state = state_

            print('The reward sum of this eposide is ' + str(reward_sum))


    # used for testing... reduce mean and the variance...
    def test_filter(self, x, mean, std, clip=10):
        x = x - mean
        x = x / (std + 1e-8)
        x = np.clip(x, -clip, clip)

        return x




























        
