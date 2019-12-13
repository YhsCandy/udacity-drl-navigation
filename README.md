# Udacity deep reinforcement learning - navigation project

## Introduction

This project train an agent to navigate and collect bananas in a large, square world. 
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
Thus, the goal of trained agent is to collect as many yellow bananas as possible while avoiding blue bananas.
Agent is considered trained if average result score in the last 100 episodes is greater or equal to 13.0.  

* Link to [original repository](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).
* Link to [training reports](Report.md).

## Environment details

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* move forward.
* move backward.
* turn left.
* turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Installation

Exclusive virtualenv is recommended:

```bash
virtualenv --python /usr/bin/python3 .venv
. .venv/bin/activate
``` 

Version that works well on MacOSX:
```bash
virtualenv -p python3 .venv
. .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

If you encounter missing tensorflow 1.7.1 dependency try this:
```bash
pip install torch numpy pillow matplotlib grpcio protobuf
pip install --no-dependencies unityagents
```

Download unity environment files:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

And/or [NoVis alternative](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) (only for Linux).

## Training

Run training with default parameters:

```bash
python3 training.py
```

All training parameters:

|Parameter|Description|Default value|
|---|---|---|
|--environment|Path to Unity environment files|Banana_Linux_NoVis/Banana.x86_64|
|--model|Path to save model|checkpoint.pth|
|--type|NN type - NoisyDueling, Dueling or Q|NoisyDueling|
|--buffer|Replay buffer type - sample or prioritized|prioritized|
|--episodes|Maximum number of training episodes|2000|
|--frames|Maximum number of frames in training episode|1000|
|--target|Desired minimal average per 100 episodes|13.0|
|--eps_start|Starting value of epsilon|1.0|
|--eps_decay|Epsilon decay factor|0.995|
|--eps_end|Minimum value of epsilon|0.01|
|--buffer_size|Replay buffer size|100000|
|--batch_size|Minibatch size|64|
|--gamma|Discount factor|0.99|
|--tau|For soft update of target parameters|0.01|
|--alpha|Prioritized buffer - How much prioritization is used (0 - no prioritization, 1 - full prioritization)|0.5|
|--beta|Prioritized buffer - To what degree to use importance weights (0 - no corrections, 1 - full correction)|0.5|
|--learning_rate|Learning rate|0.001|
|--update_every|Update every n frames|4|
|--cuda/--no_cuda|Force disable CUDA or autodetect|Autodetect|

## Testing

Run test with default parameters:

```bash
python3 testing.py
```

All testing parameters:

|Parameter|Description|Default value|
|---|---|---|
|--environment|Path to Unity environment files|Banana_Linux/Banana.x86_64|
|--model|Path to trained model|checkpoint.pth|
|--type|NN type - NoisyDueling, Dueling or Q|NoisyDueling|
|--cuda/--no_cuda|Force disable CUDA or autodetect|Autodetect|

Pretrained models (set also correct --type parameter !):

* [NoisyDueling](models/nd.pth)
* [Dueling](models/d.pth)
* [Q Network](models/q.pth)

## Future work

- Implement other RL improvements (i.e. rainbow algorithm).
- Use more hw/time to do full hyperparameter space search for the best hyperparameters of this task.

## Licensing

Code in this repository is licensed under the MIT license. See [LICENSE](LICENSE).
