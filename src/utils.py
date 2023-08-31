def sort_dict(d: dict):
    """
    Sorts a dictionary by its keys.
    :param d: The dictionary to sort
    :return: The sorted dictionary
    """
    return {k: d[k] for k in sorted(d.keys())}
