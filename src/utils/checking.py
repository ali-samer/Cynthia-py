import numpy as np
from src.utils.policy import policy as pl

def isnumpy(user_function):
    if callable(user_function) and hasattr(np, user_function.__name__):
        return True
    else:
        return False


def varcheck(var, policy): # TODO: must complete implementation
    if not callable(policy):
        pl.common.error()

    if not var:
        policy(pl.checking.msg.var, var)
