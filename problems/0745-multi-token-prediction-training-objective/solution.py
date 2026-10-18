import numpy as np

def mtp_loss(main_logits, main_targets, mtp_logits, mtp_targets, mtp_weight):
    """
    Compute the combined LM + depth-1 Multi-Token Prediction loss.

    Args:
        main_logits: array-like (N, V)
        main_targets: array-like (N,) integer token ids
        mtp_logits: array-like (N-1, V)
        mtp_targets: array-like (N-1,) integer token ids
        mtp_weight: float

    Returns:
        float: total loss rounded to 6 decimals
    """
    def cross_entropy(logits, targets):
        logits = np.asarray(logits, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.int64)

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
        target_logits = shifted[np.arange(len(targets)), targets]

        return np.mean(log_sum_exp - target_logits)
    
    loss_main = cross_entropy(main_logits, main_targets)
    loss_mtp = cross_entropy(mtp_logits, mtp_targets)

    total_loss = loss_main + mtp_weight * loss_mtp
    return round(float(total_loss), 6)

