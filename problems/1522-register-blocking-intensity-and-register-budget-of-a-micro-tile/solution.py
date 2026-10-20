import math


def intensity(tm, tn):
    """计算算术强度（每加载1个元素能执行的浮点运算次数 FLOPs/Load）"""
    return 2 * tm * tn / (tm + tn)


def registers(tm, tn):
    """计算单步所需的寄存器数量（累加器 + A、B 操作数）"""
    return tm * tn + tm + tn


def loads_per_output(tm, tn, K):
    """计算整个 K 循环中，每个输出元素分摊的内存加载次数"""
    return K * (tm + tn) / (tm * tn)


def best_tile(budget, max_side=16):
    """在寄存器预算内寻找最优分块 (tm, tn)"""
    candidates = [
        (tm, tn)
        for tm in range(1, max_side + 1)
        for tn in range(1, max_side + 1)
        if registers(tm, tn) <= budget
    ]
    # 优先级：1. intensity 最大  2. registers 最小（取负）  3. tm 最小（取负）
    return max(candidates, key=lambda t: (intensity(*t), -registers(*t), -t[0]))


def cache_block(cache_bytes, elem_bytes=4):
    """计算能同时放进缓存的 A、B、C 三个 T x T 方块的最大边长 T"""
    return math.isqrt(cache_bytes // (3 * elem_bytes))

