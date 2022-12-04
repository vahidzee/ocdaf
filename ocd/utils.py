import typing as th


def args_list_len(*args):
    return max(len(arg) if isinstance(arg, (tuple, list)) else (1 if arg is not None else 0) for arg in args)


def list_args(*args, length: th.Optional[int] = None, return_length: bool = False):
    length = args_list_len(*args) if length is None else length
    if not length:
        results = args if len(args) > 1 else args[0]
        return results if not return_length else (results, length)
    results = [([arg] * length if not isinstance(arg, (tuple, list)) else arg) for arg in args]
    results = results if len(args) > 1 else results[0]
    return results if not return_length else (results, length)
