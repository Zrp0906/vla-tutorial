# 12_02｜QLoRA 进阶

[全部教程](../../README.md) · [进阶功能路线](12_进阶功能路线.md) · [上一篇：12_01｜监督微调进阶](12_01_监督微调进阶.md) · [下一篇：12_03｜残差强化学习进阶](12_03_残差强化学习进阶.md)

## 这一部分介绍什么

阶段 07 把基础权重冻结，只训练 LoRA 适配器。QLoRA 进一步用 NF4 保存部分冻结权重。专家示范仍通过 Flow Matching 提供监督，优化器更新的是低秩适配器。 [PEFT 量化训练说明](https://huggingface.co/docs/peft/developer_guides/quantization)

## 使用说明与前提

准备好阶段 03–04 的数据和匹配的基础模型，使用装有 CUDA 版 PyTorch 的 NVIDIA 环境。下面的示例从工作区根目录执行。 [bitsandbytes 0.48.2 安装说明](https://huggingface.co/docs/bitsandbytes/v0.48.2/en/installation)

```bash
cd /Users/zrp/Desktop/lora
conda run -n embodiedai python -m pip install bitsandbytes==0.48.2
conda run -n embodiedai python smol/smol_workflow/p12_advance/12_02_train_qlora.py --run-dir smol/outputs/lesson_run --device cuda --epochs 1 --window-stride 200
```



## 输入和输出

- **输入**为基础模型、处理器、训练和验证动作窗口、训练集统计量及 LoRA 设置。
- **输出**在 `--run-dir` 指定目录下的 `models/qlora/`。每个 `epoch_N/` 保存适配器权重、PEFT 配置、训练状态和基座版本。`selected.json` 记录验证选中的检查点，`report.json` 记录训练结果。
- 现有 `Runtime` 会在 CUDA 上按相同范围重建 NF4 基座，再加载适配器。也可以把该检查点传给 12_03 的 `--base-checkpoint`。

## 原理：低位基座上训练高精度修正

对于一个加了 LoRA 的量化线性层，可以把前向计算写成

$$
y=\widehat{W}_{\mathrm{NF4}}x+\frac{\alpha}{r}BAx
$$

`Ŵ_NF4` 是计算时从 NF4 权重还原出的近似值，`A` 和 `B` 是可训练的低秩矩阵。沿用 `r=8`、`α=16` 时，适配器分支的缩放是 2。

本项目量化 VLM 文本骨干和动作专家中的 MLP 投影，在动作专家的量化 MLP 上挂 LoRA。注意力投影仍用浮点权重，因为 SmolVLA 的注意力实现会直接读取这些权重的 dtype。训练入口会在开始更新前检查 NF4 权重是否已在 CUDA 上完成量化。

## 检查清单

- 确认当前解释器能调用 CUDA，并能导入 `bitsandbytes`。
- 在 `report.json` 检查冻结权重和可训练参数的变化情况，核对 `trainable_names` 对应动作专家 MLP 中的 LoRA 参数。
- 重载时核对基座版本，确认选中的检查点可以生成有限的动作。实际显存看训练报告。

## 代码对照（可选）

运行入口是 [12_02_train_qlora.py](12_02_train_qlora.py)。[p12_qlora.py](_impl/p12_qlora.py) 负责选择量化范围、装载适配器和重载检查点。

---

[全部教程](../../README.md) · [进阶功能路线](12_进阶功能路线.md) · [上一篇：12_01｜监督微调进阶](12_01_监督微调进阶.md) · [下一篇：12_03｜残差强化学习进阶](12_03_残差强化学习进阶.md)
