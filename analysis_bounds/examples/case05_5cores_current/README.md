# case05 五核构造：3.76081948×

AI 辅助搜索得到的具体调度例子，使用仓库官方评测器验证；这是已达到的构造成绩，不是理论上界或 `npu_scheduler/` 的统一算法成绩。

| 指标 | 结果 |
| --- | ---: |
| 默认单核时间 | 18,876,779 cycles |
| 五核方案时间 | 5,019,326 cycles |
| 加速比 | 3.76081948× |
| 活动核数 | 5 |
| DDR 搬运量 | 299,892,736 bytes |

将输出网格划为三行面板，尾部两行按列分配；每条累加链保留在同核，搜索面板顺序与列方向，复用相邻面板的 B 输入。完整行列顺序见 `recipe.json`。

[plan.json](plan.json) 保存完整分核和逐子图顺序；[summary.json](summary.json) 保存原评测摘要、硬件参数及方案、数据、评测源码哈希。

从仓库根目录重新评测保存的方案，无需重跑搜索：

```bash
python -B code/multicore_cut_evaluate_problem_2.py data/case_05.json \
  analysis_bounds/examples/case05_5cores_current/plan.json \
  --config data/config.txt \
  -o /tmp/case05_evaluation.json \
  --trace-output /tmp/case05_trace.json \
  --log-output /tmp/case05_evaluation.log
```

可用保存的面板配方重新生成方案：

```bash
python -B analysis_bounds/examples/case05_5cores_current/build_plan.py -o /tmp/case05_plan.json
```

全部案例见[构造汇总](../README.md)，理论上界见[上界汇总](../../current_upper_table/README.md)。
