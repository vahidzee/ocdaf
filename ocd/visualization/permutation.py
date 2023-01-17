import typing as th

def abbriviate_consecutives(ordering: th.Iterable[int]):
    """
    Loop over the ordering (a list of unique integers) and find all consecutive sequences
    of ascending/descending numbers and replace them with the first and last number in 
    the sequence. Numbers are expected to be between 0 and len(ordering) - 1.

    For example: [0, 1, 2, 3, 4] -> ["0-4"] and [4, 3, 2, 1, 0] -> ["4-0"]
        or [0, 1, 2, 3, 10, 9, 8, 4, 5, 7, 6] -> ["0-3", "10-8", "4-5", "7-6"]

    This is useful for plotting the ordering of the permutation matrix in a more compact way.

    Args:
        ordering (th.Iterable[int]): A list of unique integers between 0 and len(ordering) - 1
    
    Returns:
        ordering_str (th.List[str]): A list of strings representing the ordering
    """
    if len(ordering) < 2:
        return ordering
    ordering, results = list(ordering), []
    ascending: bool = ordering[0] < ordering[1]
    start, start_idx = ordering[0], 0
    for i in range(1, len(ordering)):
        if ascending and ordering[i] == ordering[i-1] + 1:
            continue
        elif not ascending and ordering[i] == ordering[i-1] - 1:
            continue
        else:
            if start_idx == i - 1:
                results.append(f"{start}")
            else:
                results.append(f"{start}-{ordering[i-1]}")
            start, start_idx = ordering[i], i
            ascending = ordering[i] < ordering[i+1] if i < len(ordering) - 1 else ascending

    if start_idx < len(ordering) - 1:
        results.append(f"{start}-{ordering[-1]}")
    else:
        results.append(f"{start}")
    return results