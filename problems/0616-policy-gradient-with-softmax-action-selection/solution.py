import numpy as np

def softmax_policy_gradient_step(
    theta: np.ndarray,
    states: list,
    actions: list,
    weights: list,
    alpha: float
) -> tuple:
    N = len(states)
    if N == 0:
        return theta, []
        
    states_arr = np.array(states)
    actions_arr = np.array(actions)
    weights_arr = np.array(weights)[:, np.newaxis]
    
    # 1. 计算偏好值与 Softmax 概率
    h = theta[states_arr]
    h_max = np.max(h, axis=1, keepdims=True)
    exp_h = np.exp(h - h_max)
    probs = exp_h / np.sum(exp_h, axis=1, keepdims=True)
    
    # 2. 计算对数概率的梯度: OneHot(A) - pi
    grad = -probs 
    grad[np.arange(N), actions_arr] += 1.0
    
    # 3. 乘以权重 (Advantage 或 Return)
    weighted_grad = weights_arr * grad
    
    # 4. 更新 theta 参数 (核心修复：取整个批次的平均梯度)
    updated_theta = np.copy(theta)
    
    # 注意这里乘上了 (alpha / N) 而不是仅仅乘以 alpha
    np.add.at(updated_theta, states_arr, (alpha / N) * weighted_grad)
    
    return updated_theta, list(probs)