# Proximal Policy Optimization
This is an pytorch-version implementation of [Proximal Policy Optimisation(PPO)](https://arxiv.org/abs/1707.06347). Some function of this code is based on [ikostrikov's TRPO project](https://github.com/ikostrikov/pytorch-trpo). In this code, the actor-network and the critic-network are separately. The actions are sampled from the beta distribution which could improve the performance. The paper about this is: [The Beta Policy for Continuous Control Reinforcement Learning](https://www.ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf)

## Requirements

- python 3.5.2
- openai gym
- mujoco-python 0.5.7
- pytorch
- [pyro](http://pyro.ai/)

## Instruction to run the code
### Train your models
```bash
cd /root-of-this-code/
python train_network.py --cuda --env_name='the env you want to train' (if you have a gpu)
or
python train_network.py --env_name='the env you want to train' (if you don't have a gpu)
```
You could also try some other mujoco's environment. This code has already pre-trained two mujoco environments: `Walker2d-v1` and `Humanoid-v1`. You could try them by yourself!

### Test your models:
```bash
cd /root-of-this-code/
python demo.py --env_name='the env you want to display'

```
## Results
### Training Curve
![Training_Curve](https://github.com/TianhongDai/Reinforcement_Learning_Algorithms/blob/master/ppo/figures/result.png)
### Demo: Walker2d-v1
![Demo](https://github.com/TianhongDai/Reinforcement_Learning_Algorithms/blob/master/ppo/figures/walker2d.gif)







