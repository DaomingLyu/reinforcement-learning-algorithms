#Deep Reinforcement Learning Alogrithms
This Repository Will Implemented the Classic Deep Reinforcement Learning Algorithms.
- [ ] Deep Q-Learning Network(DQN)
- [ ] Double DQN(DDQN)
- [ ] Dueling Network Architecture
- [x] Deep Deterministic Policy Gradient(DDPG)
- [ ] Normalized Advantage Function(NAF)
- [x] Asynchronous Advantage Actor-Critic(A3C)
- [ ] Trust Region Policy Optimization(TRPO)
- [x] Proximal Policy Optimization(PPO)
- [ ] Actor Critic using Kronecker-Factored Trust Region(ACKTR)

I has already implemented three of these algorithms. I will implement the rest of algorithms and keep update them in the future.

##Something Important
In this repository, the actions are sampled from the beta distribution which could improve the performance. The paper about this is: [The Beta Policy for Continuous Control Reinforcement Learning](https://www.ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf)

However, I can't calculate the Back-Propagation of Beta Distribution's Entropy. If someone has the solution of it, please contact me.

##Requirements
- python 3.5.2
- openai gym
- mujoco-python
- pytorch
- [pyro](http://pyro.ai/)

##Instruction To Use the Code
The instruction has been introduced in each repository. In the future, I will revise them and use a common format.

##Acknowledgement:
[Ilya Kostrikov's GitHub](https://github.com/ikostrikov)

##Papers Related to the Deep Reinforcement Learning
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




 


