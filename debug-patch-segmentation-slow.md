# Debug Session: patch-segmentation-slow

## 症状
Patch 分割（regenerate）过程非常慢，第一个文件处理 >1 分钟进度条不动。

## 假设

| ID | 假设 | 状态 |
|----|------|:----:|
| A | `while` 循环条件 `available.sum() > min_remaining` 导致剩余点不足 8192 时死循环 | 待验证 |
| B | `query_ball_point` 在高密度点云上每次调用开销大 | 待验证 |
| C | 每个文件点数太多，`available.sum()` 和 `np.where` 的 O(N) 累加导致瓶颈 | 待验证 |
| D | 每个文件只有极少数 patch 能通过钢筋比例筛选，大量无用计算 | 待验证 |

## 日志

（调试服务器收集）

## 结论

待定
