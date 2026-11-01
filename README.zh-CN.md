# 多点海底调查任务路径规划

[![自动测试](https://github.com/anyang1991/multipoint-seabed-survey/actions/workflows/tests.yml/badge.svg)](https://github.com/anyang1991/multipoint-seabed-survey/actions/workflows/tests.yml)
[English](README.md) · [论文主页](https://anyang.space/publication/2017-auv-path-planning/) · [引用信息](CITATION.cff) · [版本下载](https://github.com/anyang1991/multipoint-seabed-survey/releases)

论文 **Path Planning for Multipoint Seabed Survey Mission Using Autonomous Underwater Vehicle**
的配套开源程序，作者为 Yang An、Gaofei Xu、Chunhui Xu、Hongyu Zhao、Jian Liu，发表于 OCEANS 2017 - Anchorage。

程序采用两阶段框架：通过迭代 K-means 和贪心删点确定支撑船锚点，再用蚁群算法规划满足续航与采样容量约束的闭合下潜路径。
提供 CPU/GPU 自动适配、多进程计算、可复现实验及论文插图导出。

![100点任务规划示例](examples/results/paper-style-100.png)

## 安装与运行

需要 Python 3.10 及以上：

```bash
git clone https://github.com/anyang1991/multipoint-seabed-survey.git
cd multipoint-seabed-survey
python -m pip install -e .
python -m seabed_planner plan examples/paper-style-100.json --output outputs/plan.json --plot outputs/plan.png
python -m seabed_planner validate outputs/plan.json
```

结果包含完整输入、锚点、下潜路径、搜索历史、参数、随机种子和运行信息。
图件导出为 300 dpi PNG，并附可编辑 SVG。

## 计算设置

```bash
python -m seabed_planner hardware
python -m seabed_planner plan examples/paper-style-100.json --backend cpu --workers 4
python -m seabed_planner plan examples/paper-style-100.json --backend gpu
python -m seabed_planner generate --targets 100 --seed 2017 --output outputs/mission.json
```

默认根据任务规模、可用 CPU、内存和 CUDA 设备选择计算方式及进程数。
GPU 环境可按 CUDA 版本选择安装 `.[gpu]`（CUDA 12）或 `.[gpu13]`（CUDA 13）。
[计算与性能说明](docs/performance.md)记录适配策略和实测结果。

## 模型与数据

每次下潜从所属锚点出发并返回，每个目标恰好访问一次。航行与作业总时间受续航约束，样本总数受采样容量约束。
各锚点采用 `航程 × 下潜次数²` 的联合目标，默认搜索 500 次迭代、每次 30 只蚂蚁。
模型采用恒速、平面距离与固定目标作业时间。[方法与参数](docs/method.md)

支持 JSON、CSV 和 MATLAB `.mat` 输入。目标字段为 `x,y,workload_hours,samples`；
时间单位为小时，坐标与航速使用配套的距离单位。保存的目标编号从 0 开始。
[数据来源与输入格式](examples/README.md)

## 实验复现

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m experiments.verify_claims
python -m experiments.plot_evidence
```

验证包含 12 个独立的 100 点任务、每个任务 3 个搜索种子，36 次规划全部满足约束。
相对受约束最近邻策略，联合目标值平均降低 **18.90%**。10 个 8 点固定锚点实例的 30 次搜索中，
22 次达到精确最优，平均目标差距 **0.43%**。

[复现指南](docs/reproducibility.md) · [实验结果与方案](docs/validation.zh-CN.md) · [绘图说明](docs/figures.md)

## 项目结构

`src/` 为 Python 程序，`examples/` 为输入和规划结果，`experiments/` 为验证实验及原始数据，
`benchmarks/` 为性能测试，`tests/` 为自动测试，`legacy/` 保存原 MATLAB 程序及来源记录。

## 引用与许可

使用本程序请引用论文，完整 BibTeX 见 [English README](README.md#citation-and-license)，
机器可读引用见 [CITATION.cff](CITATION.cff)。
Python 实现、测试、示例和文档采用 [MIT 许可证](LICENSE)，适用范围见 [NOTICE](NOTICE)。
问题反馈与贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。
