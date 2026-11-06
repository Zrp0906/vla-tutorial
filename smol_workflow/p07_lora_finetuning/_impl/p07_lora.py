"""
功能：
    按 SmolVLA 默认目标模块挂上 LoRA。

输入：
    无

输出：
    无

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    p06_training.py
"""
# 内部实现：由同阶段入口调用，不是独立执行步骤。

def attach_lora(policy, c):
    from peft import LoraConfig, get_peft_model
    targets = policy._get_default_peft_targets()
    cfg = LoraConfig(r=c["lora_rank"], lora_alpha=c["lora_alpha"], lora_dropout=0,
                     target_modules=targets["target_modules"], modules_to_save=[])
    wrapped = get_peft_model(policy, cfg)
    return wrapped, wrapped.get_base_model()
