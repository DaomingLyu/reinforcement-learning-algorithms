# Deep Reinforcement Learning Alogrithms
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)  
This Repository Will Implement the Classic Deep Reinforcement Learning Algorithms.
- [x] [Deep Q-Learning Network(DQN)](https://github.com/TianhongDai/reinforcement_learning_algorithms/tree/master/dqn)
- [x] [Double DQN(DDQN)](https://github.com/TianhongDai/reinforcement_learning_algorithms/tree/master/double_dqn)
- [ ] Dueling Network Architecture
- [x] [Deep Deterministic Policy Gradient(DDPG)](https://github.com/TianhongDai/reinforcement_learning_algorithms/tree/master/ddpg)
- [ ] Normalized Advantage Function(NAF)
- [x] [Asynchronous Advantage Actor-Critic(A3C)](https://github.com/TianhongDai/reinforcement_learning_algorithms/tree/master/a3c)
- [ ] Trust Region Policy Optimization(TRPO)
- [x] [Proximal Policy Optimization(PPO)](https://github.com/TianhongDai/reinforcement_learning_algorithms/tree/master/ppo)
- [ ] Actor Critic using Kronecker-Factored Trust Region(ACKTR)

I has already implemented five of these algorithms. I will implement the rest of algorithms and keep update them in the future.

## Something Important
In this repository, the actions are sampled from the beta distribution which could improve the performance. The paper about this is: [The Beta Policy for Continuous Control Reinforcement Learning](https://www.ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf)

However, I can't calculate the Back-Propagation of Beta Distribution's Entropy. If someone has the solution of it, please contact me.

## Requirements
- python 3.5.2
- openai gym
- [gym_ple](https://github.com/lusob/gym-ple)
- mujoco-py - 0.5.7
- pytorch
- [pyro](http://pyro.ai/)

## Instruction To Use the Code
The instruction has been introduced in each repository. In the future, I will revise them and use a common format.

## Acknowledgement:
- [Ilya Kostrikov's GitHub](https://github.com/ikostrikov)

## Papers Related to the Deep Reinforcement Learning
[1] [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)  
[2] [The Beta Policy for Continuous Control Reinforcement Learning](https://www.ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf)  
[3] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
[4] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)  
[5] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)  
[6] [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)  
[7] [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)  
[8] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)  
[9] [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)  
[10] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[11] [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)  




 


