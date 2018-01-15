# Deep Deterministic Policy Gradient(DDPG)
This is an pytorch-version implementation of [Deep Deterministic Policy Gradient(DDPG)](https://arxiv.org/abs/1509.02971). 

## Requirements
- python 3.5.2
- openai gym
- mujoco-python
- pytorch
- [pyro](http://pyro.ai/)

## Instruction to run the code
### Train your models
```bash
cd /root-of-this-code/
python train_network.py
or
python train_network.py --env='Walker2d-v1' (env name as you want)

```
You could also try some other mujoco's environment. This code has already pre-trained three environments: `Reacher-v1`, `Walker2d-v1` and `HalfCheetah`. You could try them by yourself!  
Some hyper-parameters of this code is used from ["Deep Reinforcement Learning that Matters"](https://arxiv.org/abs/1709.06560)  
The results of the `Walker2d-v1` and `HalfCheetah` is same as the [Paper's](https://arxiv.org/abs/1709.06560). However, it still unable to solve some other environments, like `Humanoid-v1`, but the PPO could solve it successfully. I will update the code once i find an approach to improve it.


### Test your models:
```bash
cd /root-of-this-code/
python demo.py --cuda=0
or
python demo.py --env='Walker2d-v1' --cuda=0

```






