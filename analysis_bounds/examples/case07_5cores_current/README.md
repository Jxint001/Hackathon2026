# case07 五核构造：4.27912134×

AI 辅助搜索得到的具体调度例子，使用仓库官方评测器验证；这是已达到的构造成绩，不是理论上界或 `npu_scheduler/` 的统一算法成绩。

| 指标 | 结果 |
| --- | ---: |
| 默认单核时间 | 6,340,434 cycles |
| 五核方案时间 | 1,481,714 cycles |
| 加速比 | 4.27912134× |
| 活动核数 | 5 |
| DDR 搬运量 | 75,344,608 bytes |

从卷积输入重叠关系得到空间分区，保持共写者同核；通过消费者靠近、生产者延后和等待操作前移改善流水重叠。本例的收益包含流水等待减少，并非只来自搬运量下降。

[plan.json](plan.json) 保存完整分核和逐子图顺序；[summary.json](summary.json) 保存原评测摘要、硬件参数及方案、数据、评测源码哈希。

从仓库根目录重新评测保存的方案，无需重跑搜索：

```bash
python -B code/multicore_cut_evaluate_problem_2.py data/case_07.json \
  analysis_bounds/examples/case07_5cores_current/plan.json \
  --config data/config.txt \
  -o /tmp/case07_evaluation.json \
  --trace-output /tmp/case07_trace.json \
  --log-output /tmp/case07_evaluation.log
```

全部案例见[构造汇总](../README.md)，理论上界见[上界汇总](../../current_upper_table/README.md)。
