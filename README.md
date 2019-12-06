# Udacity deep reinforcement learning - navigation project

- intro to project
- link to [original repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)
- link to [reports](Report.md)

## Environment details

- type and size of state and action space
- target criteria

## Installation

- python dependencies (tensorflow issue !)
- banana vis/novis env

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

- table with pretrained models

## Future work

- rainbow ?
- full hyperparameter space search

## Licensing

Code in this repository is licensed under the MIT license. See [LICENSE](LICENSE).
