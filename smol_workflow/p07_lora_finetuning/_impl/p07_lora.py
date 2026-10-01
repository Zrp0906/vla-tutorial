r"""按 SmolVLA 官方默认目标模块构造 PEFT LoRA 适配器。

输入：
    - 已有真实 SmolVLA policy 与配置 c；读取 lora_rank、lora_alpha。
    - 目标模块来自 policy._get_default_peft_targets()；dropout=0，modules_to_save=[]。

输出：
    - attach_lora(policy, c) 返回 PEFT 包装策略和底层 policy。
    - 本模块只创建适配器，不单独训练或写权重；共享训练循环调用 p05_model.package 保存推理包。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p07_lora_finetuning/07_01_train_lora.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本模块无命令行；attach_lora(policy, c) 是供共享训练循环调用的函数。
    - 标准配置 rank=8、alpha=16；通过 07_01 入口运行独立 LoRA 实验。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p07_lora_finetuning/07_01_train_lora.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/_impl/p06_training.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
# 内部实现：由同阶段入口调用，不是独立执行步骤。

def attach_lora(policy, c):
    from peft import LoraConfig, get_peft_model
    targets = policy._get_default_peft_targets()
    cfg = LoraConfig(r=c["lora_rank"], lora_alpha=c["lora_alpha"], lora_dropout=0,
                     target_modules=targets["target_modules"], modules_to_save=[])
    wrapped = get_peft_model(policy, cfg)
    return wrapped, wrapped.get_base_model()
