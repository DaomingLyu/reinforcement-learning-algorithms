# Asynchronous Advantage Actor-Critic (A3C)
This is an pytorch-version implementation of [Asynchronous Advantage Actor-Critic(A3C)](https://arxiv.org/abs/1602.01783). I have trained and tested them in the environment of `Pendulum-v0`(`continues control`) and `Pong-v0`(`discrete contol`). 

## Pre-Requisite
- python 3.5.2
- openai gym (atari game)
- pytorch
- [pyro](http://pyro.ai/)

## Instruction to run the code
### Train your models
```bash
cd /root-of-this-code/
python train_network_continues.py
or
python train_network_discrete.py

```

### Test your models:
```bash
cd /root-of-this-code/
python demo_continues.py
or
python demo_continues.py

```


