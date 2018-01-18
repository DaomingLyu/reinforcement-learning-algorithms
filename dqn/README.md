# Deep Q-Learning Network(DQN)
This is an pytorch-version implementation of ["Human-level control through deep reinforcement learning"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). In this project, the `FlappyBird-v0` was choosen as the training environment. So, it's very cool to try it!

## Requirements

- python 3.5.2
- openai gym
- [gym_ple](https://github.com/lusob/gym-ple)
- pytorch

## Instruction to run the code
### Train your models
```bash
cd /root-of-this-code/
python train_network.py

```
After training about 2 hours, our bird could pass about `130` pipes. So, feel free to try it!

### Test your models:
```bash
cd /root-of-this-code/
python demo.py --cuda=0

```

