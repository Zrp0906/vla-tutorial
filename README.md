# SmolVLA 工作流

二维推块仿真上的 SmolVLA 教学流程：采集示范、整理数据、评测基线、部分参数训练、LoRA 微调、对比考试、合并与 INT8 量化、HTTP 闭环。

入口文件是 `阶段_步骤_功能.py`，按编号从小到大运行。各阶段的 `_impl/` 只被入口调用。调用链见 [PYTHON_EXECUTION_FLOW.md](PYTHON_EXECUTION_FLOW.md)。



## 顺序与教程

教程按 00–11 阅读，从 [00 从零开始](smol_workflow/p00_pipeline/00_从零开始与运行路线.md) 起，不要求先懂机器学习。

完成数据准备后，可以按需运行 [12_01 SFT](smol_workflow/p12_advance/12_01_监督微调进阶.md)、[12_02 QLoRA](smol_workflow/p12_advance/12_02_QLoRA进阶.md)、[12_03 残差强化学习](smol_workflow/p12_advance/12_03_残差强化学习进阶.md) 和 [12_04 权重空间编程](smol_workflow/p12_advance/12_04_权重空间编程进阶.md)。待实现的功能见 [进阶功能路线](smol_workflow/p12_advance/12_进阶功能路线.md)。

| 顺序  | 入口                                                                  | 教程                                                                |
| --- | ------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 01  | `01_01_check_environment.py` 检查环境                                   | [01](smol_workflow/p01_environment/01_环境检查与设备选择.md)               |
| 02  | `02_01_check_simulation.py` 检查仿真                                    | [02](smol_workflow/p02_simulation/02_推块世界与运动规则.md)                |
| 03  | `03_01_collect_data.py` 采集示范                                        | [03](smol_workflow/p03_data_collection/03_专家示范与数据采集.md)           |
| 04  | `04_01_process_data.py` 整理数据                                        | [04](smol_workflow/p04_data_processing/04_数据整理与数字归一化.md)          |
| 05  | `05_01_download_assets.py` 下载模型，再 `05_02_evaluate_baseline.py` 评测基线 | [05](smol_workflow/p05_model_baseline/05_认识SmolVLA与基线评测.md)       |
| 06  | `06_01_train_partial.py` 部分参数训练                                     | [06](smol_workflow/p06_supervised_post_training/06_监督学习与流匹配训练.md) |
| 07  | `07_01_train_lora.py` LoRA 微调                                       | [07](smol_workflow/p07_lora_finetuning/07_LoRA低秩微调入门.md)          |
| 08  | `08_01_evaluate_models.py` 对比两个实验                                   | [08](smol_workflow/p08_evaluation/08_模型考试与结果判断.md)                |
| 09  | `09_01_merge_quantize.py` 合并并 INT8 量化                               | [09](smol_workflow/p09_merge_quantization/09_权重合并与INT8量化.md)      |
| 10  | `10_01_check_deployment.py` HTTP 与闭环                                | [10](smol_workflow/p10_deployment/10_HTTP服务与闭环控制.md)              |
| 11  | `11_01_summarize.py` 汇总报告                                           | [11](smol_workflow/p11_feedback_summary/11_实验总结与下一步改进.md)         |
