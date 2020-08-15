#================================================================
#
#   File name   : decors.py
#   Author      : Josiah Tan
#   Created date: 15/08/2020
#   Description : decorator file for k_armed_bandits
#
#================================================================

#================================================================

from joblib import Parallel, delayed


class repeat:
  def __init__(self, num_times = 10, n_jobs = 10, run_parallel = True, return_avg = True):
    self.num_times = num_times
    self.n_jobs = n_jobs
    self.run_parallel = run_parallel
    self.return_avg = return_avg
  
  def __call__(self, _func):
    if self.run_parallel:
      self._func = _func
      return self.parallel_decor
  
  def parallel_decor(self, *args, **kwargs):
    return_list = Parallel(n_jobs=self.n_jobs)(delayed(self._func)(*args, **kwargs) for _ in range(self.num_times))
    
    if self.return_avg:
      from numpy import mean
      return_list = mean(return_list, axis = 0)
    return return_list
