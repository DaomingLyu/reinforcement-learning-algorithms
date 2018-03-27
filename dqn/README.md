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
python train_network.py --cuda (if you have a GPU)
or
python train_network.py (if you don't have a GPU)

```
After training about 6 hours, our bird could pass about `264` pipes maximally. So, feel free to try it!

### Test your models:
```bash
cd /root-of-this-code/
python demo.py

```
## Results
### Training Curve
Although from the training plot, it could just pass about 30 pipes. However, in the demo, it could have impressive performance!
![results](https://github.com/TianhongDai/Reinforcement_Learning_Algorithms/blob/master/dqn/figures/results.png)
### Demo: Flappy-Bird
![demo](https://github.com/TianhongDai/Reinforcement_Learning_Algorithms/blob/master/dqn/figures/flappybird.gif)


