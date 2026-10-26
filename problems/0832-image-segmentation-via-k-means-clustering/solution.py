import numpy as np


def segment_image(image, k: int, max_iters: int = 100):
    # --- 0. 转 numpy，保留浮点精度 ---
    img = np.asarray(image, dtype=np.float64)
    H, W, C = img.shape

    # --- 1. 按行优先 flatten 成 (N, 3) ---
    pixels = img.reshape(-1, C)          # row-major，正确
    N = pixels.shape[0]

    # 边界保护：k 不超过像素数
    k = min(k, N)

    # --- 2. 取前 k 个像素作为初始质心 ---
    centroids = pixels[:k].copy()        # shape (k, 3)
    labels = np.zeros(N, dtype=np.int64)

    # --- 3. 迭代 K-Means ---
    for _ in range(max_iters):
        # 3a. 计算每个像素到每个质心的平方欧氏距离 (N, k)
        #     按簇循环，避免造 (N, k, 3) 大张量
        dists = np.empty((N, k), dtype=np.float64)
        for ci in range(k):
            diff = pixels - centroids[ci]          # (N, 3)
            dists[:, ci] = np.sum(diff * diff, axis=1)

        # 3b. 分配：argmin 在平局时天然取最小索引
        new_labels = np.argmin(dists, axis=1)

        # 3c. 更新质心：空簇保留旧质心
        new_centroids = centroids.copy()
        for ci in range(k):
            mask = new_labels == ci
            if mask.any():
                new_centroids[ci] = pixels[mask].mean(axis=0)

        labels = new_labels

        # 3d. 精确相等才算收敛
        if np.array_equal(new_centroids, centroids):
            break

        centroids = new_centroids

    # --- 4. 每个像素替换成它所属的质心 ---
    segmented = centroids[labels].reshape(H, W, C)

    # --- 5. 四舍五入到 4 位小数，返回嵌套 list ---
    return np.round(segmented, 4).tolist()