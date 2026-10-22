def rmsprop_update(params: list[float], grads: list[float], cache: list[float], 
                   lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
	"""
	Perform RMSProp optimization update.
	
	Args:
		params: List of parameter values
		grads: List of gradients for each parameter
		cache: List of cache values (moving average of squared gradients)
		lr: Learning rate
		beta: Decay rate for moving average
		epsilon: Small constant for numerical stability
	
	Returns:
		Tuple of (updated_params, updated_cache)
	"""
	# Your code here
	updated_cache = [beta * c + (1 - beta) * (g ** 2) for g, c in zip(grads, cache)]
    
    # 2. 用新缓存更新参数列表
    updated_params = [p - lr * g / (new_c ** 0.5 + epsilon) for p, g, new_c in zip(params, grads, updated_cache)]
    
    return updated_params, updated_cache