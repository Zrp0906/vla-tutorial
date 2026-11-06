# 05｜认识 SmolVLA 与基线评测

[全部教程](/Users/zrp/Desktop/lora/smol/README.md) · [上一篇：04｜数据整理与数字归一化](/Users/zrp/Desktop/lora/smol/smol_workflow/p04_data_processing/04_数据整理与数字归一化.md) · [下一篇：06｜监督学习与流匹配训练](/Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/06_监督学习与流匹配训练.md)

## 这一部分介绍什么

SmolVLA 可以想成三个合作的小组：看图片、读指令、决定动作。V、L、A 分别代表视觉、语言、动作。

本项目使用真实预训练 SmolVLA，而不是另写一个小网络冒充它。预训练好比“已经上过很多课”，不表示它见过这个二维游戏、知道我们的动作坐标约定。

先让原模型考试，叫“基线”：以后才知道训练有没有带来变化。官方结构介绍见 [LeRobot 的 SmolVLA 文档](https://huggingface.co/docs/lerobot/smolvla)。

## 使用说明：先获取资产，再测基线

先完成阶段 04。下载入口没有 `--run-dir`、`--config` 或 `--device` 参数：

```bash
cd /Users/zrp/Desktop/lora
conda run -n embodiedai python smol/smol_workflow/p05_model_baseline/05_01_download_assets.py
```

这个脚本可能访问网络并下载模型。若已存在资产索引，它会直接返回；索引存在不等于每个文件一定完好，缺文件时仍需排查。

接着运行基线评测：

```bash
cd /Users/zrp/Desktop/lora
conda run -n embodiedai python smol/smol_workflow/p05_model_baseline/05_02_evaluate_baseline.py --config smol/configs/standard.json --run-dir smol/outputs/lesson_run --device auto
```

## 输入和输出

共用资产保存到 `smol/models/`：

- `base/`：`lerobot/smolvla_base` 的配置和完整预训练权重。
- `processor/`：SmolVLM2 所需的文字/图片处理配置；不是又下载一套完整视觉模型权重。
- `assets.json`：模型来源和固定版本编号，便于复现。

本次实验输出：`models/base/` 中的运行信息和来源记录，以及 `evaluations/expert/`、`random/`、`base/` 中的报告、逐场 JSON 和部分 GIF。

`expert` 是规则老师；`random` 是随机乱走；`base` 是没有学本项目示范的预训练模型。三者在相同的 20 个测试场景中比较。

## 原理与公式：从观察到一串动作

$$
(a_0,a_1,\ldots,a_{49})=\pi_\theta(I,s,\ell;\varepsilon)
$$

`π` 读作“策略”，就是模型的动作决定方法；`θ` 是它的权重；`I` 是图片；`s` 是推动器位置；`ℓ` 是任务文字；`ε` 是生成动作时使用的随机噪声。右边全部是输入，左边是一串输出。

每个 `a_k=[a_{k,x},a_{k,y}]` 包含两维。比如第一项 `[0.5,0]`，经过环境的移动规则，理想情况下向右走 `0.04×0.5=0.02`。

模型一次给出 50 个动作，叫“动作块”。本项目闭环每次只执行第一个，然后重新拍照和预测，不是一口气盲走 50 步。

随机种子控制初始噪声。比较合并前后或不同模型时固定噪声，有助于分清差异来自模型还是随机抽样。

## 为什么外面两维、里面更多维

我们的小世界只需要 x、y。预训练模型内部采用更大的通用状态/动作空间；当前基础配置上限为 32 维，包装代码会按模型要求补齐，并取外部有效的两维结果。

语言先被拆成小文字单位，再变成数字编号；图片也经过模型要求的处理。这里不是“中文直接变成动作”，实际采集使用两条固定英文指令：

- `Push the red block into the blue target zone.`：把红块推到蓝色区。
- `Push the red block into the green target zone.`：把红块推到绿色区。

## 检查清单与故障边界

- 看基础模型的真实成功率，不把“预训练模型加载成功”当作“推块成功”。
- `strict=True` 要求权重与结构严格匹配；加载失败时应检查配置、库版本与资产，不要通过改成随机初始化来假装完成。
- 若遇到配置字段类型不兼容，先处理加载问题，不能继续宣称训练或评测已完成。
- 这篇介绍的是代码设计和用法，不承诺当前环境已完成一次基线评测。

## 代码对照（可选）

想看“规则写在哪里”，可以阅读 [p05_model.py](/Users/zrp/Desktop/lora/smol/smol_workflow/p05_model_baseline/_impl/p05_model.py)。这些是入口调用的内部实现，不要按文件名逐个执行；先理解教程，再看函数会更容易。

---

[全部教程](/Users/zrp/Desktop/lora/smol/README.md) · [上一篇：04｜数据整理与数字归一化](/Users/zrp/Desktop/lora/smol/smol_workflow/p04_data_processing/04_数据整理与数字归一化.md) · [下一篇：06｜监督学习与流匹配训练](/Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/06_监督学习与流匹配训练.md)
