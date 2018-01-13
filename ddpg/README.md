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

```
You could also try some other mujoco's environment. This code has already pre-trained two environments: `Pendulum-v1` and `Reacher-v1`. You could try them by yourself!

### Test your models:
```bash
cd /root-of-this-code/
python demo.py

```






