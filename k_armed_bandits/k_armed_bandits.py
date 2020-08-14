#================================================================
#
#   File name   : k_armed_bandits.py
#   Author      : Josiah Tan
#   Created date: 14/08/2020
#   Description : contains k-armed Bandit class for K-armed Bandit Notes.ipynb
#
#================================================================

#================================================================
from .configs import *
import numpy as np

class KArmedBandits:
  def __init__(self, num_actions = 10, tru_val_variance = 1, tru_val_mean = 0, sample_val_variance = 1):
    self.num_actions = num_actions 
    self.q_star = np.random.normal(tru_val_mean, tru_val_variance, num_actions) # sampling from the normal distribution to obtain q_star(a)
    self.sample_val_variance = sample_val_variance    

    self.actions = [i for i in range(num_actions)] # generates a list containing actions (integers) from 0 to num_actions - 1

  def get_reward(self, action):
      """
      get_reward: gets the reward for a certain action from the normal distribution
        parameters -- action, a integer between 0 and num_actions - 1
        returns -- action_val, a number with variance, sample_val_variance and mean, tru_action_val for a given action 
      """
      tru_action_val = self.q_star[action] # selecting the tru_action_val from an index given by action
      action_val = np.random.normal(tru_action_val, self.sample_val_variance)
      return action_val    


      
  
  


if __name__ == "__main__":
  agent = Agent()
  steps = 100
  for step in range(steps):
    action = agent.get_epsilon_greedy_action()
    reward = k_armed_bandits.get_reward(action)
    agent.update_N(action)
    agent.update_Q(reward)

a = np.array([1,2,3])
print(k)
