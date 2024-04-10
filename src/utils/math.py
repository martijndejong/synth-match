def linear_interp(lower: float, upper: float, alpha: float = 0.5) -> float:
    """
    :param lower: lower value
    :param upper: upper value
    :param alpha: value between 0 and 1 (0 returns value_a, 1 returns value_b)
    :return: linear interpolated value at alpha
    """
    return lower + alpha * (upper - lower)
