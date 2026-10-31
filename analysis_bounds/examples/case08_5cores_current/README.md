# case08 五核构造：4.45630307×

AI 辅助搜索得到的具体调度例子，使用仓库官方评测器验证；这是已达到的构造成绩，不是理论上界或 `npu_scheduler/` 的统一算法成绩。

| 指标 | 结果 |
| --- | ---: |
| 默认单核时间 | 15,419,076 cycles |
| 五核方案时间 | 3,460,060 cycles |
| 加速比 | 4.45630307× |
| 活动核数 | 5 |
| DDR 搬运量 | 183,341,280 bytes |

从卷积输入重叠关系恢复空间位置并分核；依据张量驻留、释放和流水就绪状态排序，再调整依赖层优先级及生产者位置，减少 DDR 搬运。

[plan.json](plan.json) 保存完整分核和逐子图顺序；[summary.json](summary.json) 保存原评测摘要、硬件参数及方案、数据、评测源码哈希。

从仓库根目录重新评测保存的方案，无需重跑搜索：

```bash
python -B code/multicore_cut_evaluate_problem_2.py data/case_08.json \
  analysis_bounds/examples/case08_5cores_current/plan.json \
  --config data/config.txt \
  -o /tmp/case08_evaluation.json \
  --trace-output /tmp/case08_trace.json \
  --log-output /tmp/case08_evaluation.log
```

全部案例见[构造汇总](../README.md)，理论上界见[上界汇总](../../current_upper_table/README.md)。
