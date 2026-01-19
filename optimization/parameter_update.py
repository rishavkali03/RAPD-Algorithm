import numpy as np
import random


def update_parameters(
    current_params,
    best_params,
    param_bounds,
    learning_rate=0.1,
    random_scale=0.05
):
    """
    Adaptive parameter update mechanism for RAPD.

    Parameters:
        current_params (dict): Current parameter values
        best_params (dict): Best-performing parameter values
        param_bounds (dict): Min and max bounds for parameters
        learning_rate (float): Step size toward best parameters
        random_scale (float): Random perturbation factor

    Returns:
        dict: Updated parameter values
    """

    updated_params = {}

    for param in current_params.keys():
        current_value = current_params[param]
        best_value = best_params[param]

        # Guided update toward best solution (exploitation)
        new_value = current_value + learning_rate * (best_value - current_value)

        # Adaptive random perturbation (exploration)
        search_range = param_bounds[param][1] - param_bounds[param][0]
        perturbation = random.uniform(-random_scale, random_scale) * search_range
        new_value += perturbation

        # Enforce parameter bounds
        min_val, max_val = param_bounds[param]
        new_value = np.clip(new_value, min_val, max_val)

        updated_params[param] = new_value
    
    return updated_params
