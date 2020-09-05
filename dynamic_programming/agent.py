#================================================================
#
#   File name   : agent.py
#   Author      : Josiah Tan
#   Created date: 14/08/2020
#   Description : contains Agent class for K-armed Bandit Notes.ipynb
#
#================================================================

#================================================================
#from .configs import *
import numpy as np

class Agent:
  def __init__(self, num_actions = 10, random_seed = 1):
    """
    __init__: initialisation of the Agent class
      parameters -- num_actions: the number of actions that can be take
    """
    # use random seed to allow for reproducible results
    np.random.seed(random_seed)
    
    self.num_actions = num_actions
    
    self.actions = list(range(num_actions)) # generates a list containing actions (integers) from 0 to num_actions - 1
    
    self.q_estimates = np.zeros(num_actions) # current estimate, Q(a) - the sample average
    self.N = np.zeros(num_actions) # number of times a certain action is chosen
  
  def get_action(self, epsilon = 1e-1):
    """
    get_action: a function that returns the action with the highest current estimated value or a random action depending in epsilon
    parameters: epsilon -- a number between 0 and 1
    returns: action -- an integer between 0 and num_actions - 1
    """

    if self.choose_greedy(epsilon):
      action = self.select_argmax_action()
    else:
      action = self.select_uniform_action()
    
    self.prev_action = action # store for later use
    
    return action        

  def choose_greedy(self, epsilon):
    """
    choose_greedy: returns greedy = True, with probability epsilon, otherwise greedy = False
    """
    greedy = np.random.random() > epsilon
    return greedy
  
  def select_argmax_action(self):
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
    """
    select_uniform_action: selects an action uniformly at random
    """
    return np.random.choice(self.actions)
    
  def update_N(self):
    """
    update_N: updates self.N according to what the previous action was
    """
    self.N[self.prev_action] += 1
  
  def update_Q(self, reward):
    """
    update the current estimate, q_estimates of the previous action
    """
    old_estimate = self.q_estimates[self.prev_action]
    self.q_estimates[self.prev_action] = old_estimate + 1/self.N[self.prev_action] * (reward - old_estimate)
  
