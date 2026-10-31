# case09 五核构造：3.90473944×

AI 辅助搜索得到的具体调度例子，使用仓库官方评测器验证；这是已达到的构造成绩，不是理论上界或 `npu_scheduler/` 的统一算法成绩。

| 指标 | 结果 |
| --- | ---: |
| 默认单核时间 | 6,794,313 cycles |
| 五核方案时间 | 1,740,017 cycles |
| 加速比 | 3.90473944× |
| 活动核数 | 5 |
| DDR 搬运量 | 91,388,288 bytes |

以相邻空间区域分核，再调整不同阶段的核间边界；保持共写者同核，结合张量驻留、释放和管道就绪时间调整生产与消费顺序。

[plan.json](plan.json) 保存完整分核和逐子图顺序；[summary.json](summary.json) 保存原评测摘要、硬件参数及方案、数据、评测源码哈希。

从仓库根目录重新评测保存的方案，无需重跑搜索：

```bash
python -B code/multicore_cut_evaluate_problem_2.py data/case_09.json \
  analysis_bounds/examples/case09_5cores_current/plan.json \
  --config data/config.txt \
  -o /tmp/case09_evaluation.json \
  --trace-output /tmp/case09_trace.json \
  --log-output /tmp/case09_evaluation.log
```

全部案例见[构造汇总](../README.md)，理论上界见[上界汇总](../../current_upper_table/README.md)。
