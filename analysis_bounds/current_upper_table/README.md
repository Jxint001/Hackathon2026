# AI 辅助的加速比上界分析

AI 从计算图及固定硬件约束出发推导执行时间下界，并用程序计算或复核证书。若任意合法方案的执行时间都满足 T ≥ L，则相对于默认单核时间 B 的加速比满足 S ≤ B/L。

分析保留三类方法：

- **依赖与管道负载**：[dag_bounds.py](dag_bounds.py) 放宽跨核通信、容量与额外搬运，计算关键路径、必需 DDR 工作量及 release/tail 时间窗内的资源负载界。
- **有限内存搬运下界**：[memory_bound.py](memory_bound.py) 对识别并验证为规则 Matmul 的图，利用 A/B/C 投影和容量限制给出必要 I/O 下界。
- **专项证书**：case03、case05 保留分段搬运推导及整数 DP 复核；case08 保留计算时间窗证书。[case03](certificates/case03/bound_final.md) · [case05](certificates/case05/bound_final.md) · [case08](certificates/case08/structure_upper_certificate.md)。

固定设置：L1=1,572,864 B，UB=131,072 B，共享 DDR=60 B/cycle，跨核等待=1000 cycles。分子采用同版本官方评测器的默认单核调度时间；表格保留既有计算结果，整理目录时未重新评测。

| Case | 默认单核 cycles | 2 核上界 | 3 核上界 | 4 核上界 | 5 核上界 |
| --- | ---: | ---: | ---: | ---: | ---: |
| case00 | 3,541,355 | 1.99951 | 2.99788 | 3.99532 | 4.98608 |
| case01 | 85,856 | 1.00600 | 1.00600 | 1.00600 | 1.00600 |
| case02 | 167,400 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| case03 | 6,295,267 | 2.00017 | 2.99948 | 3.99826 | 4.28194 |
| case04 | 7,080,299 | 1.99976 | 2.99894 | 3.99766 | 4.99375 |
| case05 | 18,876,779 | 1.99991 | 2.99961 | 3.99913 | 4.56417 |
| case06 | 3,540,935 | 1.99928 | 2.99752 | 3.99485 | 4.98549 |
| case07 | 6,340,434 | 2.10861 | 3.16222 | 4.21536 | 5.26803 |
| case08 | 15,419,076 | 2.55026 | 3.82521 | 5.10003 | 6.37222 |
| case09 | 6,794,313 | 2.26568 | 3.39835 | 4.53089 | 5.66332 |

表内加速比向上舍入到五位小数；精确时间下界见 [table.json](table.json)。上界不保证可达到，已构造出的成绩另见[构造例子](../examples/README.md)。默认单核调度未必最优，因此相对它的加速比上界可以超过核数。

从仓库根目录重新计算全表（会调用单核评测与内存界计算，耗时长于只读取记录）：

```bash
python -B analysis_bounds/current_upper_table/rebuild.py
```

仅复核 case03、case05 的已保存专项证书，需要 Python 标准库和 g++，无需重跑搜索：

```bash
python -B analysis_bounds/current_upper_table/certificates/case03/bound_final.py
python -B analysis_bounds/current_upper_table/certificates/case05/bound_final.py
```

脚本合并通用界与专项界；五核界也适用于核数更少的方案，因为允许空闲核。这里保留的是已记录的上界分析，不宣称已经求得最优调度。
