#================================================================
#
#   File name   : agent.py
#   Author      : Josiah Tan
#   Created date: 8/09/2020
#   Description : contains Agent and RTAgent (real time agent) class for dynamic programming 
#
#================================================================

#================================================================
#from .configs import *
import numpy as np
from IPython.display import clear_output

#agent.py
class Agent:
  def __init__(self, grid_dims, actions, rand_seed, verbose = 0):
    """
    initialises the agent class
    paramters:
      grid_dims -- a tuple (y, x)
      actions -- a dictionary mapping action integers to actions - "left", "right", "up", "down"
      rand_seed -- a "pseudo random" seed for reproducible results 
      verbose -- 1 or 0, 0 means don't print anything, 1 means print everything - useful for debugging
    """

    self.grid_dims = grid_dims
    self.actions = actions
    self.num_actions = len(actions)
    self.rand_seed = rand_seed
    self.verbose = verbose

    self.pi = np.ones((grid_dims[0], grid_dims[1], self.num_actions)) / self.num_actions
    self.V = np.zeros(grid_dims)
  
  def disp_v_pi(self, clear = True, sleep_time = 0.5, user_input = False, disp_v = True, disp_pi = True):
    """
    Prints out the matrices v and pi
    paramters:
      clear -- default is True, clears the output before printing
      sleep_time -- defualt is 0.5 seconds, the wait time after printing
      user_input -- default is False, if True wait for the user to continue the program
    """
    if clear:
      clear_output(wait=True)

    if disp_v: 
      print("v:")
      print(self.V)   

    if disp_pi:
      print("pi:")
      self.disp_pi() 

    if user_input:
      input("Press enter to continue: \n")
    else:
      sleep(sleep_time)

  def disp_pi(self, flush = False):
    greedy_actions = (self.pi == np.max(self.pi, axis = -1, keepdims=True)).astype(int)
    states = np.array([["ldur"] * self.grid_dims[1]] * self.grid_dims[0])

    for i in range(self.grid_dims[0]):
      for j in range(self.grid_dims[1]):
        optimal_actions = np.argwhere(greedy_actions[i][j]).ravel().tolist()
        state = ""
        for key, val in self.actions.items():
          if key in optimal_actions:
            state += val[0]
          else:
            state += " "
        states[i][j] = state
    print(str(states), flush = flush)

  def state_generator(self):
    for row in range(self.grid_dims[0]):
      for col in range(self.grid_dims[1]):
        state = (row, col)
        yield state

  def policy_evaluation(self, env, gamma = 0.9, theta = 0.1):
    while True:
        delta = 0
        for state in self.state_generator():
          v = self.V[state] # suprisingly, no need to use np.copy()

          self.bellman_update(env, state, gamma)
          
          delta = max(delta, np.abs(v - self.V[state]))
        if delta < theta:
          break

  def get_action_vals(self, env, state, gamma):
    actions = self.pi[state[0]][state[1]]
    action_vals = np.zeros_like(actions)
    for action in range(self.num_actions): # perform a one-step lookout
      next_state, reward = env.transition(state, action)
      action_vals[action] = reward + gamma * self.V[next_state] # calculate a bootstrapped return
    return action_vals

  def bellman_update(self, env, state, gamma):
    if env.is_terminal(state): # check if the state is terminal
        return
    action_vals = self.get_action_vals(env, state, gamma)
    actions = self.pi[state[0]][state[1]]
    self.V[state] = np.dot(action_vals,actions)
    
  def policy_iteration(self, env, gamma = 0.9, theta = 0.1):
    policy_stable = False
    while not policy_stable:

      self.disp_v_pi(user_input = True)

      self.policy_evaluation(env, gamma, theta)
      policy_stable = self.policy_improvement(env, gamma, theta)
      
  def policy_improvement(self, env, gamma = 0.9, theta = 0.1):
    policy_stable = True
    for state in self.state_generator():
      old_action = np.copy(self.pi[state[0]][state[1]])
      self.q_greedify_policy(env, state, gamma)
      if not np.array_equal(old_action, self.pi[state[0]][state[1]]):
        policy_stable = False
    return policy_stable
  
  def q_greedify_policy(self, env, state, gamma):
    if env.is_terminal(state): # check if the state is terminal
        return

    action_vals = self.get_action_vals(env, state, gamma)
    greedy_actions = action_vals == np.max(action_vals)
    self.pi[state[0]][state[1]] = greedy_actions / np.sum(greedy_actions)

  def value_iteration(self, env, gamma = 0.9, theta = 0.1):
    while True:
      delta = 0
  
      self.disp_v_pi(user_input = True, disp_pi = False)

      for state in self.state_generator():
        v = self.V[state]
        self.bellman_optimality_update(env, state, gamma)
        delta = max(delta, np.abs(v - self.V[state]))
      if delta < theta:
        break
    for state in self.state_generator():
      self.q_greedify_policy(env, state, gamma)

    self.disp_v_pi()
    
  def value_iteration_v2(self, env, gamma = 0.9, theta = 0.1):
    while True:
      delta = 0
      
      self.disp_v_pi(user_input = True)

      for state in self.state_generator():
        v = self.V[state]
        self.q_greedify_policy(env, state, gamma)
        self.bellman_update(env, state, gamma)
        delta = max(delta, np.abs(v - self.V[state]))
      if delta < theta:
        break
    for state in self.state_generator():
      self.q_greedify_policy(env, state, gamma)

    self.disp_v_pi()

  def bellman_optimality_update(self, env, state, gamma):
    if env.is_terminal(state): # check if the state is terminal
        return

    action_vals = self.get_action_vals(env, state, gamma)

    self.V[state] = np.max(action_vals)
 
class RTAgent(Agent):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
  def real_time_dynamic_programming(self, env, gamma, horizon):
    np.random.seed(self.rand_seed)
    state = env.get_rand_state()
    for t in range(horizon):
      
      self.disp_v_pi(sleep_time = 0.1)

      self.q_greedify_policy(env, state, gamma)
      self.bellman_update(env, state, gamma)

      #self.bellman_optimality_update(env, state, gamma)
      
      
      actions = self.pi[state[0]][state[1]]
      action = np.random.choice(range(self.num_actions), p = actions)
      state = env.get_state(state, action)

      if env.is_terminal(state):
        state = env.get_rand_state()


if __name__ == "__main__":
  class TestAgent:

    @staticmethod
    def test_v_pi():
      agent = Agent(grid_dims, actions, rand_seed)
      print(agent.V)
      print(agent.pi)

    @staticmethod
    def test_state_generator():
      agent = Agent(grid_dims, actions, rand_seed)
      for row, col in agent.state_generator():
        print(row, col, end= ', ')

    @staticmethod
    def test_policy_evaluation():
      agent = Agent(grid_dims, actions, rand_seed)
      env = GridWorld(grid_dims, actions, terminal_states, rand_seed)
      agent.policy_evaluation(env, gamma = 1, theta = 0.1)
      print(agent.V)

    @staticmethod
    def test_policy_iteration():
      agent = Agent(grid_dims, actions, rand_seed)
      env = GridWorld(grid_dims, actions, terminal_states, rand_seed)
      env.rewards[3,1:] = -10 # 5x5
      env.rewards[1,0:4] = -10 # 5x5

      print("rewards")
      print(env.rewards)
      print("environment")
      env.disp(env.get_rand_state())
      input("Press Enter to continue \n")

      agent.policy_iteration(env, gamma = 1, theta = 0.1)

      agent.policy_evaluation(env, gamma = 1, theta = 0.1)

      print("Evaluation of Optimal Policy")
      print(agent.V)

    @staticmethod
    def test_value_iteration():
      agent = Agent(grid_dims, actions, rand_seed)
      env = GridWorld(grid_dims, actions, terminal_states, rand_seed)
      env.rewards[3,1:] = -10 # 5x5
      env.rewards[1,0:4] = -10 # 5x5

      print("rewards")
      print(env.rewards)
      print("environment")
      env.disp(env.get_rand_state())
      input("Press Enter to continue \n")

      agent.value_iteration(env, gamma = 1, theta = 0.1)
      
    @staticmethod
    def test_value_iteration_v2():
      agent = Agent(grid_dims, actions, rand_seed)
      env = GridWorld(grid_dims, actions, terminal_states, rand_seed)
      env.rewards[3,1:] = -10 # 5x5
      env.rewards[1,0:4] = -10 # 5x5

      print("rewards")
      print(env.rewards)
      print("environment")
      env.disp(env.get_rand_state())
      input("Press Enter to continue \n")
      
      agent.value_iteration_v2(env, gamma = 1, theta = 0.1)

    @staticmethod
    def test_real_time_dynamic_programming():
      agent = RTAgent(grid_dims, actions, rand_seed)
      env = GridWorld(grid_dims, actions, terminal_states, rand_seed)
      env.rewards[3,1:] = -10 # 5x5
      env.rewards[1,0:4] = -10 # 5x5

      print("rewards")
      print(env.rewards)
      print("environment")
      env.disp(env.get_rand_state())
      input("Press Enter to continue \n")

      agent.real_time_dynamic_programming(env, gamma = 1, horizon = 300)
  
  #TestAgent.test_policy_evaluation()
  #TestAgent.test_state_generator()         
  #TestAgent.test_policy_iteration()
  #TestAgent.test_value_iteration()
  #TestAgent.test_value_iteration_v2()
  TestAgent.test_real_time_dynamic_programming()
