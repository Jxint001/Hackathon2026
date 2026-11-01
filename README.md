# NPU 多核调度优化

Hackathon 题目二（场景 B）的 Conv/Matmul 多核调度方案，C++ 实现位于 `npu_scheduler/`。

原数学建模题目见[《通用神经网络处理器下的多核调度问题》（PDF）](docs/通用神经网络处理器下的多核调度问题.pdf)。

项目展示海报见 [Poster（PDF）](docs/Poster.pdf)。

## 目录结构

```text
huawei/
├── README.md                    # 项目说明、源码阅读顺序与构建运行
├── npu_scheduler/               # C++17 方案生成程序
│   ├── CMakeLists.txt           # Linux/Windows 构建配置
│   ├── README.md                # 程序说明
│   ├── include/                # 头文件：数据结构与函数声明
│   │   ├── graph.hpp / plan.hpp / scheduler.hpp
│   │   ├── initial_plan.hpp / plan_search.hpp / plan_evaluator.hpp
│   │   ├── json_parser.hpp / thread_pool.hpp
│   │   ├── search/             # 搜索组件声明
│   │   └── evaluation/         # 内部模拟数据结构
│   └── src/                    # .cpp 实现文件
│       ├── main.cpp / scheduler.cpp
│       ├── graph.cpp / plan.cpp / initial_plan.cpp
│       ├── plan_search.cpp / plan_evaluator.cpp
│       ├── search/             # 候选生成、阶段控制和搜索策略
│       └── evaluation/         # 任务构造、调度与事件模拟
├── code/                       # 官方评测工具
├── data/                       # 官方计算图与评测配置
├── docs/                       # 赛题、Poster、算法、评测与提交说明
├── demo/                       # 两核简化图的算法演示与固定结果回放
└── analysis_bounds/            # AI 辅助的加速比上界分析与构造例子
```

AI 辅助的加速比上界分析与高加速比构造见 [analysis_bounds](analysis_bounds/README.md)。

## 建议阅读顺序

以下阅读顺序均针对 `npu_scheduler/` 下的源码。先读 [src/scheduler.cpp](npu_scheduler/src/scheduler.cpp)，了解“初始分组 → 初始解 → 搜索”的总流程；再读 [include/plan.hpp](npu_scheduler/include/plan.hpp) 和 [include/graph.hpp](npu_scheduler/include/graph.hpp)，理解数据；随后读 [src/search/search_controller.cpp](npu_scheduler/src/search/search_controller.cpp) 的 `run()`、`phase()`、`consider()`。某个策略的细节按下表查找。

下表中的 `.cpp` 位于 `npu_scheduler/src/`，`.hpp` 位于 `npu_scheduler/include/`，子目录名称保持一致。

| 文件 | 职责 |
| --- | --- |
| `main.cpp` | 参数解析、输入读取、方案写出和退出码 |
| `graph.hpp/.cpp` | 输入图、依赖关系、共享写者与短链初始分组 |
| `plan.hpp/.cpp` | 方案、成绩、统计，以及提交 JSON 序列化 |
| `scheduler.cpp` | 串接初始化与后续搜索 |
| `initial_plan.cpp` | 枚举核心数、遍历顺序和分核偏好，快速筛选后完整复核 |
| `plan_search.cpp` | 完整搜索的外部入口 |
| `search/search_controller.*` | 阶段安排、候选接受、缓存复用、最终方案返回 |
| `search/search_state.hpp` | 一个搜索起点的最优解、计数、缓存和起点池 |
| `search/candidate_generator.*` | 结构摘要、分核投影、资源下界、生产者延后、消费者前移、就绪排序 |
| `search/candidate_evaluator.*` | 候选哈希、评估映射构造、完整方案构造 |
| `search/structure_analysis.cpp` | 从张量共享关系推断空间位置 |
| `search/grid_search.cpp` | 网格分核、子图粒度、对齐条带、容量分块 |
| `search/spatial_search.cpp` | 空间分区与多管道重叠候选 |
| `search/order_search.cpp` | 枚举核内顺序调整，提交给统一评估入口 |
| `search/chain_search.cpp` | 整链、半链与双目标迁移 |
| `search/multistart_search.cpp` | 空间和矩阵起点选择、独立搜索及结果合并 |
| `plan_evaluator.cpp` | 内部性能模型的四步入口 |
| `evaluation/task_builder.cpp` | 合并同核任务、构造输入输出和跨核搬运 |
| `evaluation/local_schedule.cpp` | 按依赖和子图顺序生成核内指令顺序 |
| `evaluation/memory_schedule.cpp` | 换出换入、容量释放和再分配依赖 |
| `evaluation/global_simulation.cpp` | 模拟管道执行、跨核等待与共享 DDR |
| `evaluation/model.hpp` | 任务、缓冲区、内存状态、事件状态；仅供模型内部使用 |
| `json_parser.hpp` / `thread_pool.hpp` | 自实现 JSON 解析器 / 固定线程池 |

## 构建与运行

采用 C++17，没有第三方库，无需联网。Linux 使用系统线程库。

在仓库根目录运行，Linux（GCC 9.4+，CMake 3.16+）：

```bash
cmake -S npu_scheduler -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/npu_sched "data/case_00.json" -n 4 -o "out/plan.json"
```


核心数默认 4；未指定输出时写到输入同目录的 `<case>_multicore_res.json`。支持长选项 `--cores`、`--output` 以及 `--option=value`。退出码：0 成功、1 内部错误、2 参数错误、3 输入错误、4 写出错误、5 无合法方案。每进程最多 10 个线程。

比赛提交与验收规范见[提交要求文档](docs/提交要求文档.md)。
