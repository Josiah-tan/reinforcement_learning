#================================================================
#
#   File name   : agent.py
#   Author      : Josiah Tan
#   Created date: 14/08/2020
#   Description : contains Agent class for K-armed Bandit Notes.ipynb
#
#================================================================

#================================================================
from .config import *
import numpy as np

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
    
    self.actions = list(range(num_actions)) # generates a list containing actions (integers) from 0 to num_actions - 1
    
    self.q_estimates = np.zeros(num_actions) # current estimate, Q(a) - the sample average
    self.N = np.zeros(num_actions) # number of times a certain action is chosen
  
  def get_epsilon_greedy_action(self, epsilon = 1e-1):
    """
    get_epsilon_greedy_action: a function that returns the action with the highest current estimated value or a random action depending in epsilon
    """

    if is_greedy(epsilon):
      action = self.select_argmax_action()
    else:
      action = self.select_uniform_action()

    return action        

  def is_greedy(self, epsilon):
    """
    is_greedy: returns greedy drawn from probability, epsilon of greedy (boolean)
    """
    greedy = np.random.random() > epsilon
    return greedy
  
  def select_argmax_action(self, run):
    """
    select_argmax_action: selects the greedy action, with ties broken randomly
    returns: the greedy action
    """
    top = float('-inf')
    ties = []
    for action, q_estimate in zip(self.actions, self.q_estimates):
      if top < q_estimate:
          top, ties = q_estimate, [action] 
      elif top == q_estimate:
        ties.append(action)

    greedy_action = np.random.choice(ties)
    return greedy_action
  
  def select_uniform_action(self):
    return np.random.choice(self.actions)
    
  def update_N(self, action):
    self.N[action] += 1
  
  def update_Q(self, action, reward):
    old_estimate = self.q_estimates[action]
    self.q_estimates[action] = old_estimate + 1/N * (reward - old_estimate)
  
