from analyse.eval_ import compute_apfd

def test_1():
    # 假设有 5 条数据，id 为 1..5，其中 2,4 是 fault
    ground_truth_faults = [2, 4]

    # 情况一：fault 都排在最前面（最理想）
    ranked_best = [2, 4, 1, 3, 5]
    print("APFD (best):", compute_apfd(ground_truth_faults, ranked_best))

    # 情况二：完全随机
    ranked_mid = [1, 2, 3, 4, 5]
    print("APFD (mid):", compute_apfd(ground_truth_faults, ranked_mid))

    # 情况三：fault 都排在最后（最差）
    ranked_worst = [1, 3, 5, 2, 4]
    print("APFD (worst):", compute_apfd(ground_truth_faults, ranked_worst))


if __name__ == "__main__":
    test_1()