# Trust Region Policy Optimization
This is an pytorch-version implementation of [Trust Region Policy Optimisation(TRPO)](https://arxiv.org/abs/1502.05477). Some function of this code is based on [ikostrikov's TRPO project](https://github.com/ikostrikov/pytorch-trpo) and [John Schulman's code](https://github.com/joschu/modular_rl). In this code, the actor-network and the critic-network are separately. 

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
python train_network.py --env_name='the env you want to train' 
```
You could also try some other mujoco's environment. This code has already pre-trained two mujoco environments: `Walker2d-v1` . You could try it by yourself!

### Test your models:
```bash
cd /root-of-this-code/
python demo.py --env_name='the env you want to display'

```
## Results
### Training Curve
![Training_Curve](https://github.com/TianhongDai/Reinforcement_Learning_Algorithms/blob/master/trpo/figures/result.png)
### Demo: Walker2d-v1
![Demo](https://github.com/TianhongDai/Reinforcement_Learning_Algorithms/blob/master/trpo/figures/walker2d.gif)



