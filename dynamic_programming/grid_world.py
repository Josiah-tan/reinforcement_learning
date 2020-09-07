#================================================================
#
#   File name   : grid_world.py
#   Author      : Josiah Tan
#   Created date: 7/09/2020
#   Description : contains grid_world class for Dynamic Programming
#
#================================================================

#================================================================
#from .configs import *
import numpy as np

#gridworld.py
class GridWorld:
  def __init__(self, grid_dims, actions, terminal_states, rand_seed):
    self.grid_dims = grid_dims
    self.actions = actions
    self.terminal_states = terminal_states
    self.num_actions = len(actions)
    self.rewards = -np.ones(self.grid_dims).astype(int)
    self.rand_seed = rand_seed
    #self.terminal_rewards()
    #self.s = [None, None] # current state

  def terminal_rewards(self):
    pass
    # interesting idea, didn't use it

  def get_rand_state(self):
    np.random.seed(self.rand_seed)
    state = (np.random.choice(self.grid_dims[0]), np.random.choice(self.grid_dims[1]))
    return state

  def transition(self, state, action):
    next_state = self.get_state(state, action)
    reward = self.get_reward(next_state)  
    return (next_state, reward)

  def get_reward(self, state):
    reward = self.rewards[state]
    return reward

  def get_state(self, state, action):
    next_state = list(state) # prevent pass by reference
    if self.is_terminal(state):
      return tuple(next_state)

    action_str = self.actions[action]
    if action_str == "left":
      if next_state[1] > 0:
        next_state[1] -= 1
    elif action_str == "down":
      if next_state[0] < self.grid_dims[0] - 1:
        next_state[0] += 1
    elif action_str == "up":
      if next_state[0] > 0:
        next_state[0] -= 1
    elif action_str == "right":
      if next_state[1] < self.grid_dims[1] - 1:
        next_state[1] += 1
    return tuple(next_state)

  def is_terminal(self, state):
    for terminal in zip(*self.terminal_states):
      if terminal == state:
        return True
    return False

  def disp(self, state, flush = False):
    """
    disp for printing the environment via env.disp(state)
    """
    states = np.array([['| |']*self.grid_dims[1]]*self.grid_dims[0])
    states[self.terminal_states] = '|T|'
    states[state] = '|A|'
    print(str(states), flush = flush)


if __name__ == "__main__":
  #testing the environment
  env = GridWorld(grid_dims, actions, terminal_states, rand_seed)
  state = env.get_rand_state()
  np.random.seed(rand_seed)
  for i in range(25):
    sleep(0.3)
    clear_output(wait=True)
    state, reward= env.transition(state, action = np.random.choice(4))
    env.disp(state, flush = True)
    print(f"state = {state}, reward = {reward}")
