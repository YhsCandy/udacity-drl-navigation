from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque

from dqn_agent import Agent

print("Testing")

env_test = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

# get the default brain
brain_name = env_test.brain_names[0]
brain = env_test.brains[brain_name]

# initialize agent
env_info = env_test.reset(train_mode=False)[brain_name] # reset the environment
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = agent.act(state)                      # select an action
    env_info = env_test.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))

env_test.close()
