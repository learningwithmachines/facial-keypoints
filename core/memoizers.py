from functools import *
import atexit
import cupy as cp

from tempfile import mkdtemp
from joblib import Memory, Parallel, delayed

location=None

ocvh_libmemo = Memory(location=location,verbose=0,mmap_mode='r')

_gmemos_ = []

def ocvh_smemo(for_each_device=False):
    """
        Makes a function memoizing the result for each argument and device.
        This decorator provides automatic memoization of the function result.

    Args:
        for_each_device (bool): If True, it memoizes the results for each
        device. Otherwise, it memoizes the results only based on the
        arguments.

    """

    def decorator(f):
        global _gmemos_
        memo = {}
        _gmemos_.append(memo)
        none = object()

        @wraps(f)
        def ret(*args, **kwargs):
            arg_key = (args, frozenset(kwargs.items()))
            if for_each_device:
                arg_key = (cp.cuda.Device().id, arg_key)
            result = memo.get(arg_key, none)
            if result is none:
                result = f(*args, **kwargs)
                memo[arg_key] = result
            return result

        return ret

    return decorator

smemo = partial(ocvh_smemo,True)
jlibmemo  = partial(ocvh_libmemo.cache,)

@atexit.register
def ocvh_memoclear():
    global _gmemos_
    for memo in _gmemos_:
        memo.clear()