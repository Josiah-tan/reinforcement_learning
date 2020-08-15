#================================================================
#
#   File name   : utils.py
#   Author      : Josiah Tan
#   Created date: 15/08/2020
#   Description : utils file for k_armed_bandits
#
#================================================================

#================================================================
import numpy as np
def sample_k_armed_bandits(cls, num_samples = 1000, plot = True):
  """
  sample_k_armed_bandits: samples the k_armed_bandit model with num_samples for each action and plots it using a basic boxplot
  """
  sample_action_vals = np.zeros((cls.num_actions, num_samples))
  for action in cls.actions:
    for num_sample in range(num_samples):
      sample_action_vals[action, num_sample] = cls.get_reward(action)
  if plot:
    plt.boxplot(sample_action_vals.T)
