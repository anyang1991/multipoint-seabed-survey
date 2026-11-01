# 多点海底调查任务路径规划

这是我们论文 **Path Planning for Multipoint Seabed Survey Mission Using Autonomous Underwater
Vehicle**（Yang An、Gaofei Xu、Chunhui Xu、Hongyu Zhao、Jian Liu，OCEANS 2017 - Anchorage）
的开源程序。基于作者提供的 `海阔天空程序.ZIP` 小修改版 MATLAB 程序，重新整理为 Python 项目。

[论文主页](https://anyang.space/publication/2017-auv-path-planning/) ·
[论文 PDF](https://anyang.space/files/IEEE-OCEANS-2017-1.pdf) · [English](README.md)

## 论文结论复核

新增 12 个独立的 100 点任务、每个 3 个搜索种子的对照实验：36 次全部满足资源约束，
相对受约束最近邻策略，联合目标值平均降低 **18.90%**。10 个 8 点实例的 30 次搜索中，
22 次达到固定锚点精确最优，平均差距 **0.43%**。

复核围绕两阶段任务规划框架展开，覆盖锚点减少、目标可达、资源约束和完整任务代价；
蚁群算法作为路径求解工具。详见 [完整复核报告与对照图](docs/validation.zh-CN.md)。

## 安装与运行

```bash
git clone https://github.com/anyang1991/multipoint-seabed-survey.git
cd multipoint-seabed-survey
python -m pip install -e ".[test]"
python -m seabed_planner hardware
python -m seabed_planner plan examples/legacy-15.json --output outputs/legacy.json --plot outputs/legacy.png
python -m seabed_planner validate outputs/legacy.json
python -m seabed_planner plan examples/paper-style-100.json --output outputs/survey.json --plot outputs/survey.png
```

需要 Python 3.10 及以上。基础安装即可使用 CPU。CUDA 12 环境可额外安装
`python -m pip install -e ".[gpu]"`；CUDA 13 环境使用 `python -m pip install -e ".[gpu13]"`。
同一环境只安装一种 CuPy，具体要求参见 [CuPy 官方说明](https://docs.cupy.dev/en/stable/install.html)。

绘图采用统一的论文插图风格，展示路径与锚点、相对收敛进展、逐次下潜的航行/作业时间和采样容量。
PNG 为 300 dpi，同时导出可编辑的 SVG；已有 JSON 结果可直接重新绘图。[绘图说明](docs/figures.md)

## 算法与改进

1. 根据续航、航速和各目标点作业时间，用迭代 K-means 和贪心删点选择支撑船锚点。
2. 对每个锚点负责的全部目标点，用蚁群算法优化多次下潜的闭合路径。
3. 独立复核每个目标恰好访问一次，并逐次检查下潜的往返航行时间、作业时间和采样容量。

已补齐原修改版路径搜索遗漏的作业时间扣除、出发时可行性检查和完整任务规划。
默认采用论文的距离与下潜次数联合目标 `总航程 × 下潜次数^κ`（κ=2），而原修改版主要优化航程。
锚点初始化采用稀疏的非空网格，避免为整个海域创建逐像素矩阵；聚类移动和删点均检查实际覆盖约束。
算法仍属于启发式方法，不保证全局最优。[具体差异](docs/method.md)有逐项说明。

## 加速与自动适配

- 距离矩阵、最近锚点分配及蚁群候选筛选均使用批量运算。
- 自动读取当前进程可用 CPU 数、物理核心、内存和可用 CUDA 设备。
- 不同锚点可分配到多个进程并行计算，限制每个进程内部的数值库线程数。
- 小任务自动保持串行；可通过 `--workers 4` 指定并行规模，或 `--serial` 强制串行。
- 可通过 `--backend gpu` 使用 CuPy/CUDA。GPU 执行蚁群数值计算；锚点选择仍使用 CPU。
  蚂蚁路径构建合并到 CUDA 计算内核中，减少逐步调用和同步开销。
  自动模式对 GPU 使用保守阈值，避免小数据因启动和调度开销变慢。
- CPU 串行与并行使用稳定的逐锚点随机种子；相同环境与参数可得到相同结果。
  GPU 随机数实现不同，路径可以不同，但同样必须通过可行性检查。

```bash
python -m seabed_planner plan examples/paper-style-100.json --backend cpu --workers 4
python -m seabed_planner plan examples/paper-style-100.json --backend gpu
python benchmarks/benchmark.py --targets 100 --iterations 100 --repeats 3 --gpu
python benchmarks/benchmark_colony.py
python -m pytest -q
```

[性能说明](docs/performance.md)记录实测条件和结果。距离矩阵的微基准加速倍数与完整规划加速倍数分开报告；
原 MATLAB 修改版与新实现的约束逻辑不同，因此不把它们的运行时间直接作为等价算法加速比较。

本机实测：100 点、500 次迭代的完整规划由 CPU 多进程加速约 **3.02 倍**；
40 点、2048 只蚂蚁的单锚点算例由 GPU 加速约 **15.96 倍**。
小批次的普通算例仍优先使用 CPU。各项结果均保留三次计时和可行性验证记录。

## 数据和结果

- `examples/legacy-15.json`：原压缩包 `temp.mat` 的 15 点数据，续航 8 小时、航速 7.2 距离单位/小时。
- `examples/paper-style-100.json`：新生成的 100 点带状分布示例，续航 40 小时、航速 1 海里/小时、容量 3。
  原论文绘图使用的 100 点随机数据不在该压缩包中，因此本示例不是论文图的逐点重现。
- 支持 JSON、CSV 和原始 MATLAB `.mat` 输入；CSV 字段为 `x,y,workload_hours,samples`。
- 坐标采用统一单位的平面直角坐标，航速单位与之配套；经纬度应先投影。时间单位为小时。
- 保存的 JSON 包含完整输入、参数、随机种子、路径、收敛曲线、硬件和耗时；路径中的目标编号从 0 开始。
  每条路径隐含从所属锚点出发并返回。
- `legacy/matlab/` 保留原文件供对照，`CITATION.cff` 提供论文引用信息。

新 Python 实现、测试、示例和文档采用 MIT 许可证。原 MATLAB 文件保留原有声明，详见
[LICENSE](LICENSE) 和 [来源说明](legacy/README.md)。
