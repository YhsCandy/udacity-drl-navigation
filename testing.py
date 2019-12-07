from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import argparse

from agent import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--environment', type=str, help='Path to Unity environment files',
                        default='Banana_Linux/Banana.x86_64')
    parser.add_argument('--model', type=str, help='Path to trained model',
                        default='checkpoint.pth')
    parser.add_argument('--type', type=str, help='NN type - NoisyDueling, Dueling or Q',
                        default='NoisyDueling')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    print('Testing')
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.environment)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # initialize agent
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    agent = Agent(state_size=state_size, action_size=action_size, seed=0,
                  training=False, args=args)

    # load nn
    agent.qnetwork_local.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state)                      # select an action
        env_info = env.step(action)[brain_name]   # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break

    print("Score: {}".format(score))

    env.close()
