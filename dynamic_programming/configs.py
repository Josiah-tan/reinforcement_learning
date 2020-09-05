#================================================================
#
#   File name   : configs.py
#   Author      : Josiah Tan
#   Created date: 14/08/2020
#   Description : configuration file for k_armed_bandits
#
#================================================================

#================================================================

k = 10 # number of actions to choose
tru_val_variance, tru_val_mean= 1, 0  # variance and mean for initialisation of q_star(a)
sample_val_variance = 1 # variance for sample action-val distribution for every action
steps = 100 # number of steps
runs = 100
