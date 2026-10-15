def array_drills(op, *args):
    # 1. 除自身以外数组的乘积 (LeetCode 238) - O(n) 时间, O(1) 额外空间
    if op == "product":
        nums = args[0]
        n = len(nums)
        out = [1] * n
        prefix = 1
        for i in range(n):
            out[i] = prefix
            prefix *= nums[i]
        suffix = 1
        for i in range(n - 1, -1, -1):
            out[i] *= suffix
            suffix *= nums[i]
        return out

    # 2. 三数之和 (LeetCode 15) - 排序 + 双指针 O(n^2)
    elif op == "3sum":
        nums = sorted(args[0])
        n = len(nums)
        res = []
        for i in range(n - 2):
            if nums[i] > 0:  # 最小数大于 0，后面不可能凑出 0
                break
            if i > 0 and nums[i] == nums[i - 1]:  # 跳过重复的第一个数
                continue
            l, r = i + 1, n - 1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:
                    l += 1
                elif s > 0:
                    r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    l += 1
                    r -= 1
                    # 跳过重复的第二、第三个数
                    while l < r and nums[l] == nums[l - 1]:
                        l += 1
                    while l < r and nums[r] == nums[r + 1]:
                        r -= 1
        return res

    # 3. 寻找峰值 (LeetCode 162) - 二分查找 O(log n)
    elif op == "peak":
        nums = args[0]
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] < nums[mid + 1]:
                l = mid + 1  # 处于上坡，峰值一定在右侧
            else:
                r = mid      # 处于下坡，峰值在左侧或就是 mid
        return l

    # 4. 最大交换 (LeetCode 670) - 贪心记录最后出现位置 O(d)
    elif op == "maxswap":
        num = args[0]
        digits = list(str(num))
        # 记录每个数字 0-9 最后一次出现的下标
        last = {int(d): i for i, d in enumerate(digits)}
        for i, d in enumerate(digits):
            # 从 9 到当前数字+1 倒序找：有没有比当前数字更大、且出现在更靠后位置的数
            for larger in range(9, int(d), -1):
                if last.get(larger, -1) > i:
                    j = last[larger]
                    digits[i], digits[j] = digits[j], digits[i]
                    return int("".join(digits))
        return num

    # 5. 合并两个有序数组 (LeetCode 88) - 逆向双指针 O(m + n)
    elif op == "merge":
        nums1, m, nums2, n = args
        i, j, k = m - 1, n - 1, m + n - 1
        while j >= 0:
            if i >= 0 and nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        return nums1
