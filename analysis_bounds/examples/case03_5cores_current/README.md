# case03 五核构造：3.56654973×

AI 辅助搜索得到的具体调度例子，使用仓库官方评测器验证；这是已达到的构造成绩，不是理论上界或 `npu_scheduler/` 的统一算法成绩。

| 指标 | 结果 |
| --- | ---: |
| 默认单核时间 | 6,295,267 cycles |
| 五核方案时间 | 1,765,086 cycles |
| 加速比 | 3.56654973× |
| 活动核数 | 4 |
| DDR 搬运量 | 105,578,496 bytes |

将 8×24 个输出累加链划为 4×4 块，分给四个活动核，第五核为空；每块按累加深度交错执行，再调整块顺序与层内顺序以复用输入。

[plan.json](plan.json) 保存完整分核和逐子图顺序；[summary.json](summary.json) 保存原评测摘要、硬件参数及方案、数据、评测源码哈希。

从仓库根目录重新评测保存的方案，无需重跑搜索：

```bash
python -B code/multicore_cut_evaluate_problem_2.py data/case_03.json \
  analysis_bounds/examples/case03_5cores_current/plan.json \
  --config data/config.txt \
  -o /tmp/case03_evaluation.json \
  --trace-output /tmp/case03_trace.json \
  --log-output /tmp/case03_evaluation.log
```

全部案例见[构造汇总](../README.md)，理论上界见[上界汇总](../../current_upper_table/README.md)。
