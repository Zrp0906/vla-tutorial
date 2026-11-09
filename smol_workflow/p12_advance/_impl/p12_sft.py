"""
功能：
    加载基础模型，保留动作相关参数的训练状态，并解冻指定的 VLM 文本层。

输入：
    项目 models/base/config.json
    项目 models/base/model.safetensors
    项目 models/processor/ 下的处理器文件
    项目 models/assets.json

输出：
    返回调整冻结范围后的模型供公共训练循环使用。

运行：
    作为依赖导入，不单独运行。

参数：
    c：实验配置，其中 sft_text_layers 决定解冻层数
    dev：模型设备

被谁依赖：
    12_01_train_sft.py
"""
# 内部实现：由同阶段入口调用，不是独立执行步骤。


def load_sft(c, dev):
    from smol_workflow.p05_model_baseline._impl.p05_model import load_base

    policy = load_base(c, dev)
    backbone = policy.model.vlm_with_expert
    layers = backbone.get_vlm_model().text_model.layers
    count = c["sft_text_layers"]
    if not 1 <= count < len(layers):
        raise ValueError(f"--text-layers 必须在 1–{len(layers) - 1} 之间")
    backbone.train_expert_only = False
    policy.config.train_expert_only = False
    # 最后一层的文本输出不再传给后续动作层，保留其冻结状态。
    for layer in layers[-count-1:-1]:
        layer.requires_grad_(True)
    return policy, None
