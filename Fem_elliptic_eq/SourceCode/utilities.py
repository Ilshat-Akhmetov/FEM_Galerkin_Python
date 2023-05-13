from typing import Callable


def get_func(func: Callable, deriv_var: str = None, deriv_deg: int = None):
    if deriv_var is None:
        return func["func_val"]
    else:
        return func[deriv_var][deriv_deg - 1]
