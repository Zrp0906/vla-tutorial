"""
功能：
    构建 NF4 基座并挂上 LoRA，或重建同样的量化基座并重载已有适配器。

输入：
    项目 models/base/config.json
    项目 models/base/model.safetensors
    项目 models/processor/ 下的处理器文件
    项目 models/assets.json
    CUDA 环境中的 bitsandbytes
    重载时使用检查点目录里的适配器和 provenance.json

输出：
    返回量化后的策略和 PEFT 适配器对象。

运行：
    作为依赖导入，不单独运行。

参数：
    c：包含 LoRA 参数的实验配置
    dev：CUDA 设备
    checkpoint：可选的 QLoRA 检查点目录

被谁依赖：
    12_02_train_qlora.py
    p05_model.py
"""
# 内部实现：由同阶段入口调用，不是独立执行步骤。


def load_qlora(c, dev, *, checkpoint=None):
    if dev.type != "cuda":
        raise ValueError("QLoRA 入口需要可用的 NVIDIA CUDA")
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("请在 CUDA 环境安装 bitsandbytes==0.48.2") from exc

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from smol_workflow.p00_pipeline._impl.p00_common import ROOT, read_json
    from smol_workflow.p05_model_baseline._impl.p05_model import load_base

    if checkpoint is not None:
        revision = read_json(checkpoint / "provenance.json")["base_revision"]
        if revision != read_json(ROOT / "models" / "assets.json")["base"]["revision"]:
            raise ValueError("QLoRA 适配器与本地基座版本不匹配")

    policy = load_base(c, torch.device("cpu"))
    backbone = policy.model.vlm_with_expert
    targets, quantized = [], []
    # 注意力代码直接读取权重 dtype，因此只量化 MLP 的投影。
    for name, layer in list(backbone.named_modules()):
        if not isinstance(layer, torch.nn.Linear) or not name.endswith(
            (".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj")
        ):
            continue
        replacement = bnb.nn.Linear4bit(
            layer.in_features, layer.out_features, bias=layer.bias is not None,
            compute_dtype=torch.float16, compress_statistics=True, quant_type="nf4",
        )
        replacement.load_state_dict(layer.state_dict())
        replacement.requires_grad_(False)
        parent, _, child = name.rpartition(".")
        backbone.get_submodule(parent).add_module(child, replacement)
        quantized.append(replacement)
        if name.startswith("lm_expert."):
            targets.append(f"model.vlm_with_expert.{name}")
    if not targets:
        raise RuntimeError("未找到 SmolVLA 动作专家的 MLP 目标层")

    policy.to(dev)
    if any(layer.weight.quant_state is None for layer in quantized):
        raise RuntimeError("bitsandbytes 没有在 CUDA 上完成 NF4 权重量化")
    policy.is_loaded_in_4bit = True
    if checkpoint is not None:
        adapter = PeftModel.from_pretrained(policy, str(checkpoint), is_trainable=False)
    else:
        prepare_model_for_kbit_training(policy, use_gradient_checkpointing=False)
        adapter = get_peft_model(policy, LoraConfig(
            r=c["lora_rank"], lora_alpha=c["lora_alpha"], target_modules=targets,
        ))
    return adapter.get_base_model(), adapter
