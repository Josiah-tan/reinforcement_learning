#================================================================
#
#   File name   : k_armed_bandits.py
#   Author      : Josiah Tan
#   Created date: 14/08/2020
#   Description : contains the modules for k_armed_bandits for K-armed Bandit Notes.ipynb
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

class Agent:
  def __init__(self, num_actions = 10):
    """
    __init__: initialisation of the Agent class
      parameters -- num_actions: the number of actions that can be taken
                 -- runs: the number of runs to average over
                 -- tru_val_variance: variance of the true values q_star(a)
                 -- tru_val_mean: mean of the true values q_star(a)
                 -- sample_val_variance: variance of sampled rewards 
    """
    self.num_actions = num_actions
    
    self.actions = [i for i in range(num_actions)] # generates a list containing actions (integers) from 0 to num_actions - 1
    
    self.q_estimate = np.zeros(num_actions) # current estimate, Q(a) - the sample average
    self.N = np.zeros(num_actions) # number of times a certain action is chosen
    
  def p_choose_greedy(self, epsilon):
    """
    p_choose_greedy: returns greedy drawn from probability, epsilon of greedy (boolean)
    """
    greedy = np.random.random() > epsilon
    return greedy
  
  def select_argmax_action(self, run):
    """
    select_argmax_action: selects the greedy action for all runs, with ties broken randomly
    returns: a list containing the greedy action for all runs
    """
    top = float('-inf')
    ties = []
    for num_action in range(self.num_actions):
      if top < self.q_estimate[num_action, run]:
          top, ties = self.q_estimate[num_action, run], [run] 
        elif top == self.q_estimate[num_action, run]:
          ties.append(run)
    
    greedy_action = np.random.choice(ties)
    return greedy_action
  
  def select_uniform_action(self):
    return np.random.choice(self.actions)
    
  
  def get_epsilon_greedy_action(self, epsilon = 1e-1):
    """
    get_epsilon_greedy_action: a function that returns the action array with the highest current estimated value for all runs
    """
    action = np.zeros(runs)
    
    for run, greedy in enumerate(self.p_choose_greedy(epsilon)):
      if greedy:
        action[run] = self.select_argmax_action(run)
      else:
        action[run] = self.select_uniform_action()
    
    return action      
      
  
  


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
