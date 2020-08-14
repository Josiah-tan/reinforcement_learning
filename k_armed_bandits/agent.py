#================================================================
#
#   File name   : agent.py
#   Author      : Josiah Tan
#   Created date: 14/08/2020
#   Description : contains Agent class for K-armed Bandit Notes.ipynb
#
#================================================================

#================================================================

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
