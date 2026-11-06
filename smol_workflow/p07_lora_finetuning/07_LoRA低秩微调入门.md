# 07｜LoRA 低秩微调入门

[全部教程](/Users/zrp/Desktop/lora/smol/README.md) · [上一篇：06｜监督学习与流匹配训练](/Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/06_监督学习与流匹配训练.md) · [下一篇：08｜模型考试与结果判断](/Users/zrp/Desktop/lora/smol/smol_workflow/p08_evaluation/08_模型考试与结果判断.md)

## 这一部分介绍什么

普通微调像重新修改练习册正文；LoRA 像给正文旁边贴一张小便签，先不改正文，只学便签内容。

这里“小”指可训练参数少，不表示不需要基础模型，也不保证一定更快、更省全部运行内存。正式解释见 [PEFT 的 LoRA 文档](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)。

## 使用说明与前提

按流程完成阶段 06 后运行本阶段，便于公平对照。但是 LoRA 重新从同一份基础权重开始，不接着部分微调的检查点训练。

```bash
cd /Users/zrp/Desktop/lora
conda run -n embodiedai python smol/smol_workflow/p07_lora_finetuning/07_01_train_lora.py --config smol/configs/standard.json --run-dir smol/outputs/lesson_run --device auto
```

仍然训练 2 轮，使用相同的训练/验证划分和流匹配损失。标准配置 `lora_rank=8`、`lora_alpha=16`，学习率 `0.0001`。

## 输入和输出

输入：基础模型、数据窗口、统计量和 LoRA 设置。

输出在 `models/lora/`：每轮适配器模型包、训练状态、`progress.json`、`selected.json`、`report.json`。

适配器可以理解为“便签文件”。单独拿走它不能完整推理；还需要匹配的基础模型、配置和统计量。

## 原理：把一个大改动拆成两个小表

“矩阵”先理解为数字表格。“低秩”表示用少数中间通道表达修改；不是把正确答案打折。

设原权重 `W` 有 `d_out` 行、`d_in` 列。`d_in` 是输入数字个数，`d_out` 是输出数字个数。LoRA 加两个小表：

$$
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},\qquad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r}
$$

`r` 是中间通道数，也叫 rank；`ℝ` 表示表格里装的是实数。`×` 在这里是“几行、几列”，不是把两个数字相乘的计算结果。

先由 `A` 把输入变成 `r` 个中间数字，再由 `B` 变成输出。两个步骤连起来就是 `BA`；写成 `AB` 则顺序和形状可能不对。

## 公式一：原答案加上小修正

$$
W'=W+\frac{\alpha}{r}BA,\qquad
y=Wx+\frac{\alpha}{r}BAx
$$

`W'` 是修正后的权重；`x` 是输入；`y` 是输出；`α` 控制修正强度。这里省略线性层原有偏置，只解释 LoRA 新增的部分。

本项目 `r=8`、`α=16`，所以缩放 `α/r=2`。一个一维玩具例子：`W=3`、`A=0.1`、`B=0.2`、缩放为 2、`x=4`，则原输出 12，修正为 `2×0.2×0.1×4=0.16`，新输出 12.16。这只是说明运算，不是模型实测权重。

## 公式二：为什么训练的数字少了

$$
P_{\mathrm{full}}=d_{\mathrm{out}}d_{\mathrm{in}},\qquad
P_{\mathrm{LoRA}}=r(d_{\mathrm{in}}+d_{\mathrm{out}})
$$

`P` 表示参数个数。一个 100×100 的表有 10000 个数字；若用 `r=2` 的 LoRA，则两个小表共 `2×(100+100)=400` 个数字，只是原表的 4%。这是单层算例，不是整个 SmolVLA 的压缩率。

“少训练数字”不等于“基础权重消失”。图片编码和中间计算还会占内存，不能用上面的 4% 推断整机内存也只有 4%。

## 本项目具体改哪里

目标模块来自模型提供的默认 PEFT 配置，不是给所有层随便加便签。当前默认包括动作专家注意力中的部分投影，以及状态/动作等相关投影。

冻结基础参数，只训练适配器；`dropout=0` 表示本实验没有额外随机丢弃这部分连接。模块清单和可训练参数数量仍以实际报告为准，库版本变化可能影响默认目标。

## 检查清单与常见误解

- 验证基础参数未改变、适配器参数确实改变，并检查验证选出的检查点。
- 本阶段与部分微调是两个独立实验，不是把两种训练自动叠加。
- rank 更大有更多表达空间，但不保证成功率更高。
- 合并是阶段 09 的事；这一步保存的主要是适配器。

## 代码对照（可选）

想看“规则写在哪里”，可以阅读 [p07_lora.py](/Users/zrp/Desktop/lora/smol/smol_workflow/p07_lora_finetuning/_impl/p07_lora.py)。这些是入口调用的内部实现，不要按文件名逐个执行；先理解教程，再看函数会更容易。

---

[全部教程](/Users/zrp/Desktop/lora/smol/README.md) · [上一篇：06｜监督学习与流匹配训练](/Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/06_监督学习与流匹配训练.md) · [下一篇：08｜模型考试与结果判断](/Users/zrp/Desktop/lora/smol/smol_workflow/p08_evaluation/08_模型考试与结果判断.md)
