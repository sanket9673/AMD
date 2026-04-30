def normalize_metric(val: float, min_val: float, max_val: float, lower_is_better: bool = True) -> float:
    """
    Normalizes a metric to [0, 1].
    If lower_is_better=True, min_val becomes 1.0 and max_val becomes 0.0.
    If lower_is_better=False, min_val becomes 0.0 and max_val becomes 1.0.
    """
    if max_val == min_val:
        return 1.0  # If all values are the same, they all get max score
        
    # Clamp value just in case
    val = max(min_val, min(val, max_val))
    
    norm = (val - min_val) / (max_val - min_val)
    
    if lower_is_better:
        return 1.0 - norm
    return norm
