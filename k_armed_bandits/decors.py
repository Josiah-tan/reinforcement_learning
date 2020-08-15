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
from copy import deepcopy

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

  

class override_method:
  def __init__(self, cls, store_original = False):
    """
    __init__: basically overrides any method in a specified class
    parameters -- cls: class
               -- store_original: whether or not you should store the original function that is to be overwritten
    """
    self.cls = cls
    self.store_original = store_original
  def __call__(self, _func):
    #self._func = _func
    if self.store_original:
      _func.original = deepcopy(self.cls.__dict__[_func.__name__]) # storing the old method in case you need it for later
    setattr(self.cls, _func.__name__, _func) # setting the old method to the new user defined method
    
    return self.wrapper    

  def wrapper(self, *args, **kwargs):
    return self.cls.__dict__[_func.__name__](*args, **kwargs) # returning the result of calling the new method

if __name__ == '__main__':
  test_case = "test_override_method"
  if test_case == "test_override_method":

    class Happy:
      def happy(self):
        print("hi")
      def sad(self, is_sad = True):
        if is_sad:
          print('sad')

    print(Happy().happy)
    print(Happy().sad)

    dict(Happy.__dict__)

    @override_method(cls = Happy)
    def happy(self):
      print("sad")

    @override_method(cls = Happy, store_original = True)
    def sad(self, is_sad = True):
      self.sad.original(self, is_sad)
      if not is_sad:
        print("not sad")
    print(Happy().happy)
    print(Happy().sad)
    print(Happy().sad.__dict__)
    print(Happy().sad())
    print(Happy().sad(is_sad = False))
