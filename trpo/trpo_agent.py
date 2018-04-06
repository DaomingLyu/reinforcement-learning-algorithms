import torch
import numpy as np
from torch.autograd import Variable
import models
import os
from running_state import ZFilter
import pyro
import pyro.distributions as dist
from utils import conjugated_gradient
from utils import line_search
from utils import set_flat_params_to

"""
This module was used to define the TRPO algoritms

Date: 2018-04-03

Author: Tianhong Dai

"""

class trpo_brain:
    def __init__(self, args, env):
        # define the arguments and environments...
        self.args = args
        self.env = env
        # define the num of inputs and num of actions
        num_inputs = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        # define the model save dir...
        self.saved_path = 'saved_models/' + str(self.args.env_name) + '/'
        # check the path
        if not os.path.exists(self.saved_path):
            os.mkdir(self.saved_path)
        # define the networks...
        self.policy_network = models.Policy(num_inputs, num_actions)
        self.value_network = models.Value(num_inputs)
        # define the optimizer
        self.optimizer_value = torch.optim.Adam(self.value_network.parameters(), lr=self.args.value_lr, weight_decay=self.args.l2_reg)
        # init the filter...
        self.running_state = ZFilter((num_inputs,), clip=5)

    # start to train the network...
    def train_network(self):
        num_of_iteration = 0
        while True:
            # reset the brain memory...
            brain_memory = []
            batch_reward = 0
            for ep in range(self.args.batch_size):
                # reset the state....
                state = self.env.reset()
                state = self.running_state(state)
                reward_sum = 0
                # time steps in one game...
                for time_step in range(self.args.max_timestep):
                    state_tensor = torch.Tensor(state).unsqueeze(0)
                    # put it into the Variable
                    state_tensor = Variable(state_tensor)
                    # get the parameters of the action's beta distribution....
                    action_mean, action_std = self.policy_network(state_tensor)
                    # convert the variable from cuda to cpu, because cuda can not convert to numpy...
                    action_selected = self._action_selection(action_mean, action_std)
                    action_selected_cpu = action_selected.data.numpy()[0]
                    # input the action and get the feedback...
                    action_actual = action_selected_cpu.copy()
                    #action_actual = -1 + action_actual * 2
                    state_, reward, done, _  = self.env.step(action_actual)
                    # if reach the maxmimum time step, turn to terminal
                    if time_step >= self.args.max_timestep - 1:
                        done = True
                    # sum of the reward
                    reward_sum += reward
                    # process the state...
                    state_ = self.running_state(state_)
                    # store the transitions...
                    brain_memory.append((state, action_selected_cpu, reward, done))
                    # if finish jump into the next episode 
                    if done:
                        break
                    state = state_
                # sum the reward for each batch....
                batch_reward += reward_sum
            # now we could update the network...
            loss_value, loss_policy = self._update_the_network(brain_memory)
            batch_reward = batch_reward / self.args.batch_size 
            if num_of_iteration % self.args.display_interval == 0:
                print('The iteration number is ' + str(num_of_iteration) + ' and the reward mean is ' + str(batch_reward) + \
                                            ', the value_loss is ' + str(loss_value) + ', the policy_loss is ' + str(loss_policy))
            if num_of_iteration % self.args.save_interval == 0:
                path_model = self.saved_path + 'model.pt'
                torch.save([self.policy_network.state_dict(), self.running_state], path_model)
                print('=====================================================')
                print('models has been saved!!!')
                print('=====================================================')
            num_of_iteration += 1

    # start to update the network...
    def _update_the_network(self, brain_memory):
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

        # put the tensor into the Variable
        state_batch_tensor = Variable(state_batch_tensor)
        action_batch_tensor = Variable(action_batch_tensor)

        # calculate the estimated return value of the state...
        returns, advantages = self._calculate_the_discounted_reward(reward_batch_tensor, done_batch, state_batch_tensor)

        # start to update the value network...
        loss_value = self._update_value_network(returns, state_batch_tensor)
        
        # start to update the policy network
        loss_policy = self._update_policy_network(state_batch_tensor, advantages, action_batch_tensor)

        return loss_value.data.numpy()[0], loss_policy.data.numpy()[0]


    # selection the actions according to the gaussian distribution...
    def _action_selection(self, action_mean, action_std, exploration=True):
        if exploration:
            action = dist.normal(action_mean, action_std)
        else:
            action = dist.Normal(action_mean, action_std).analytic_mean()
        
        return action

    # calculate the discounted reward...
    def _calculate_the_discounted_reward(self, reward_batch_tensor, done_batch, state_batch_tensor):
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
                returns[index, 0] = reward_batch_tensor[index] + self.args.gamma * previous_return
                advantages[index, 0] = returns[index, 0] - predicted_value.data[index, 0]

            previous_return = returns[index, 0]

        # normalize the advantages...
        advantages = (advantages - advantages.mean()) / advantages.std()    

        return returns, advantages
    
    # update the value network...
    def _update_value_network(self, returns, state_batch_tensor):
        targets = Variable(returns)
        for _ in range(self.args.value_update_step):
            predicted_value = self.value_network(state_batch_tensor)
            # calculate the loss...
            loss_value = (predicted_value - targets).pow(2).mean()
            self.optimizer_value.zero_grad()
            loss_value.backward()
            # update the network
            self.optimizer_value.step()

        return loss_value
    
    # update the policy network...
    def _update_policy_network(self, state_batch_tensor, advantages, action_batch_tensor):
        action_mean_old, action_std_old = self.policy_network(state_batch_tensor)
        old_normal_dist = dist.Normal(action_mean_old, action_std_old)
        # get the corresponding probability from the beta distribution...
        old_action_prob = old_normal_dist.batch_log_pdf(action_batch_tensor)
        old_action_prob = old_action_prob.detach()
        action_mean_old = action_mean_old.detach()
        action_std_old = action_std_old.detach()
        # here will calculate the surrogate object
        surrogate_loss = self._get_surrogate_loss(state_batch_tensor, advantages, action_batch_tensor, old_action_prob)
        # compute the surrogate gradient -> g, Ax = g, where A is the Fisher Information Matrix...
        surrogate_grad = torch.autograd.grad(surrogate_loss, self.policy_network.parameters())
        flat_surrogate_grad = torch.cat([grad.view(-1) for grad in surrogate_grad]).data
        # use the conjugated gradient to calculate the scaled direction(natrual gradient)
        natural_grad = conjugated_gradient(self._fisher_vector_product, -flat_surrogate_grad, 10, \
                                            state_batch_tensor, action_mean_old, action_std_old)
        # calculate the scale ratio...
        non_scale_kl = 0.5 * (natural_grad * self._fisher_vector_product(natural_grad, state_batch_tensor, \
                                                            action_mean_old, action_std_old).sum(0, keepdim=True)) 
        scale_ratio = torch.sqrt(non_scale_kl / self.args.max_kl)
        final_natural_grad = natural_grad / scale_ratio[0]

        # calculate the expected improvement rate...
        expected_improvement = (-flat_surrogate_grad * natural_grad).sum(0, keepdim=True) / scale_ratio[0]
        # get the flat param ...
        prev_params = torch.cat([param.data.view(-1) for param in self.policy_network.parameters()])
        # start to do the line search..
        success, new_params = line_search(self.policy_network, self._get_surrogate_loss, prev_params, \
                                final_natural_grad, expected_improvement, state_batch_tensor, advantages, action_batch_tensor, old_action_prob)
        # set the params to the models...
        set_flat_params_to(self.policy_network, new_params)

        return surrogate_loss

    
    # calculate the surrogate object loss...
    def _get_surrogate_loss(self, state_batch_tensor, advantages, action_batch_tensor, old_action_prob):
        # if not needs the calculation of gradient, set volatile as True
        advantages = Variable(advantages)
        action_mean, action_std = self.policy_network(state_batch_tensor)
        normal_dist = dist.Normal(action_mean, action_std)
        new_action_prob = normal_dist.batch_log_pdf(action_batch_tensor)
        # calculate the loss...
        surrogate_loss = -torch.exp(new_action_prob - old_action_prob) * advantages
        surrogate_loss = surrogate_loss.mean()

        return surrogate_loss
    
    # calculate the kl_divergence of the two normal distribution
    def _get_kl(self, state_batch_tensor, action_mean_old, action_std_old):
        action_mean, action_std = self.policy_network(state_batch_tensor)
        # calculate the kl...
        kl = torch.log(action_std / action_std_old) + 0.5 * (action_std_old.pow(2) + (action_mean_old - action_mean).pow(2)) / \
                                            (2 * action_std.pow(2)) - 0.5

        return kl.sum(1, keepdim=True)

    # the product of the Fisher Information Matrix and natrue gradient... -> Ax
    def _fisher_vector_product(self, v, state_batch_tensor, action_mean_old, action_std_old):
        kl = self._get_kl(state_batch_tensor, action_mean_old, action_std_old)
        # calculate the mean kl...
        kl = kl.mean()
        # start to calculate the second order gradient of the KL
        kl_grads = torch.autograd.grad(kl, self.policy_network.parameters(), create_graph=True)
        flat_kl_grads = torch.cat([grad.view(-1) for grad in kl_grads])

        kl_v = (flat_kl_grads * Variable(v)).sum()
        kl_second_grads = torch.autograd.grad(kl_v, self.policy_network.parameters())
        flat_kl_second_grads = torch.cat([grad.contiguous().view(-1) for grad in kl_second_grads]).data
        # add the damping coefficient
        flat_kl_second_grads = flat_kl_second_grads + self.args.damping * v

        return flat_kl_second_grads

# ========================================= Test the Network =============================================
    
    # Here was to test the network...
    def test_network(self):
        model_path = 'saved_models/' + self.args.env_name + '/model.pt'
        # load the models
        policy_model, fiter_model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.policy_network.load_state_dict(policy_model)
        self.policy_network.eval()
        while True:
            state = self.env.reset()
            state = self._test_filter(state, fiter_model.rs.mean, fiter_model.rs.std)
            reward_sum = 0
            for _ in range(10000):
                self.env.render()
                state_tensor = torch.Tensor(state).unsqueeze(0)
                state_tensor = Variable(state_tensor)
                # get the actions...
                action_mean, action_std = self.policy_network(state_tensor)
                action_selected = self._action_selection(action_mean, action_std, exploration=False)
                action_selected_numpy = action_selected.data.numpy()[0]
                # input the action into the env...
                state_, reward, done, _ = self.env.step(action_selected_numpy)
                reward_sum += reward
                state_ = self._test_filter(state_, fiter_model.rs.mean, fiter_model.rs.std)
                if done:
                    break
                state = state_

            print('The reward sum of this eposide is ' + str(reward_sum))

    # used for testing... reduce mean and the variance...
    def _test_filter(self, x, mean, std, clip=10):
        x = x - mean
        x = x / (std + 1e-8)
        x = np.clip(x, -clip, clip)

        return x

