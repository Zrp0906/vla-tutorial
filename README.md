# SmolVLA 工作流索引

执行入口使用“阶段_步骤_功能.py”，按数字从小到大执行。阶段目录中的 _impl/ 是被入口调用的内部代码，不是独立运行步骤。

## 先自检，再选择一种运行方式

1. 00_01_check_structure.py：检查编号、导入、参数与实际调度顺序，不下载、不训练。
2. 00_02_run_pipeline.py：自动执行下表的全部步骤；也可不用总入口，按下表逐个执行。

总入口与逐步运行是两种方式，不要在总入口完成后又重复执行同一运行目录的采集或训练步骤。

## 主流程的实际执行顺序

| 顺序 | 文件 | 做什么 |
| --- | --- | --- |
| 01_01 | 01_01_check_environment.py | 检查现有 embodiedai 环境 |
| 02_01 | 02_01_check_simulation.py | 检查仿真与专家控制 |
| 03_01 | 03_01_collect_data.py | 采集成功示范 |
| 04_01 | 04_01_process_data.py | 转换数据、划分轨迹和计算统计量 |
| 05_01 | 05_01_download_assets.py | 获取固定版本的预训练模型与处理器 |
| 05_02 | 05_02_evaluate_baseline.py | 基础模型、随机策略和专家策略评测 |
| 06_01 | 06_01_train_partial.py | 部分参数监督后训练 |
| 07_01 | 07_01_train_lora.py | 从基础模型独立进行 LoRA 微调 |
| 08_01 | 08_01_evaluate_models.py | 比较两个训练实验 |
| 09_01 | 09_01_merge_quantize.py | 合并 LoRA、INT8 存储量化及复测 |
| 10_01 | 10_01_check_deployment.py | 新进程重载、HTTP 与仿真闭环检查 |
| 11_01 | 11_01_summarize.py | 汇总真实报告与失败场景 |

总入口中的 STEPS 按上表排列，子进程按此顺序执行，某一步失败后立即停止。模型下载位于 05_01，不再在环境检查之前隐式执行。仅执行阶段 05 时，也会先运行 05_01，再运行 05_02。

## 文件夹组织

    smol/
    ├── README.md
    ├── PLAN.md
    ├── requirements.txt
    ├── configs/                          # standard.json 默认；smoke.json 仅为可选诊断
    ├── smol_workflow/
    │   ├── p00_pipeline/
    │   │   ├── 00_01_check_structure.py
    │   │   ├── 00_02_run_pipeline.py
    │   │   └── _impl/                    # 公共工具与调度实现
    │   ├── p01_environment/
    │   │   ├── 01_01_check_environment.py
    │   │   └── _impl/
    │   ├── p02_simulation/
    │   │   ├── 02_01_check_simulation.py
    │   │   └── _impl/
    │   ├── p03_data_collection/
    │   │   ├── 03_01_collect_data.py
    │   │   └── _impl/
    │   ├── p04_data_processing/
    │   │   ├── 04_01_process_data.py
    │   │   └── _impl/
    │   ├── p05_model_baseline/
    │   │   ├── 05_01_download_assets.py
    │   │   ├── 05_02_evaluate_baseline.py
    │   │   └── _impl/
    │   ├── p06_supervised_post_training/
    │   │   ├── 06_01_train_partial.py
    │   │   └── _impl/
    │   ├── p07_lora_finetuning/
    │   │   ├── 07_01_train_lora.py
    │   │   └── _impl/
    │   ├── p08_evaluation/
    │   │   ├── 08_01_evaluate_models.py
    │   │   └── _impl/
    │   ├── p09_merge_quantization/
    │   │   ├── 09_01_merge_quantize.py
    │   │   └── _impl/
    │   ├── p10_deployment/
    │   │   ├── 10_01_check_deployment.py
    │   │   └── _impl/
    │   └── p11_feedback_summary/
    │       ├── 11_01_summarize.py
    │       └── _impl/
    ├── models/                           # 预训练资产：assets.json、base/、processor/
    └── outputs/<run>/                    # 运行时生成，未执行时可以不存在
        ├── config.json
        ├── stages/
        ├── data/
        ├── models/                       # 实验检查点，不是上面的预训练资产目录
        ├── evaluations/
        └── reports/

标准包文件 __init__.py 在上图省略。内部模块仍保留 pXX_ 前缀标明归属，但全部放在 _impl/；它们由函数调用驱动，不存在统一的逐文件执行顺序，也不应手动逐个运行。

LoRA 与部分参数微调共用训练循环；模型评测也复用同一评测器。这些共享调用不改变上表中实验的执行顺序。

## 命令

以下命令在项目根目录 lora/ 执行，使用已有 embodiedai 环境。

结构自检：

    conda run -n embodiedai python smol/smol_workflow/p00_pipeline/00_01_check_structure.py

只查看真实执行顺序（不创建运行目录，不启动任何流程步骤）：

    conda run -n embodiedai python smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py --dry-run

完整流程（会下载、采集和训练）：

    conda run -n embodiedai python smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py --config smol/configs/standard.json --run-dir smol/outputs/standard_run --device auto

单独检查环境：

    conda run -n embodiedai python smol/smol_workflow/p01_environment/01_01_check_environment.py --config smol/configs/standard.json --run-dir smol/outputs/standard_run --device auto

分段运行仍使用 --from-stage 与 --to-stage，参数指的是第一层阶段编号。下载脚本 05_01 无需运行目录参数；其余主流程入口共用 --config、--run-dir 和 --device。默认 standard 仍是 50 条成功示范、40 条训练/10 条验证、20 个测试场景、两项训练各 2 epoch。

## 本次调整与验证边界

本次仅更改 smol 的入口编号、内部代码位置、引用、调度顺序与文档。pi0、模型资产、数据与训练权重未修改；没有安装依赖或启动真实采集、下载、训练和评测。

结构检查还会模拟子进程调用，确认文件排序等于实际执行排序、阶段 05 内部顺序正确、只查看顺序没有副作用，以及上游失败后停止。模拟检查与参数检查不代表真实模型训练和闭环效果通过，实际完成情况仍以 outputs/<run>/stages/ 和报告为准。

此次检查在 embodiedai 环境实测通过：14 个执行入口编号唯一，15 个内部模块导入正常，入口参数、实际调度顺序、阶段 05 的先后顺序、dry-run 无执行副作用与失败即停检查均通过。

普通系统进程用于规避此前在受限进程中遇到的 OpenMP 共享内存限制。此前旧布局留下的缓存备份仍在 /private/tmp/smol-layout-backup.ILreKj。
