def sort_dict(d: dict):
    """
    Sorts a dictionary by its keys.
    :param d: The dictionary to sort
    :return: The sorted dictionary
    """
    return {k: d[k] for k in sorted(d.keys())}


def powerset(lst):
    """
    Returns the powerset of a list, but without the empty set.
    :param lst: The list to get the powerset of
    :return: The powerset of the list as a list of lists
    """
    if not lst:
        return [[]]

    subsets = powerset(lst[:-1])
    new_subsets = [subset + [lst[-1]] for subset in subsets]
    result = subsets + new_subsets

    # remove empty set from result
    result.pop(0)

    return result
