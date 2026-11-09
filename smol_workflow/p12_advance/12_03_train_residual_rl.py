"""
功能：
    冻结 SmolVLA，用仿真奖励训练残差策略，再与基础策略对照。

输入：
    <run>/config.json（默认读取，也可用 --config 指定原始配置）
    <run>/data/raw/manifest.json 中的独立测试场景
    <run>/data/stats.json
    项目 models/base/config.json
    项目 models/base/model.safetensors
    项目 models/processor/ 下的处理器文件
    项目 models/assets.json
    可选的完整微调模型或适配器目录

输出：
    <run>/models/residual_rl/residual.safetensors
    <run>/models/residual_rl/report.json
    默认使用基础模型时，按需写入 <run>/models/base/ 下的运行信息和来源信息

运行：
    独立运行。

参数：
    --run-dir：已完成阶段 03–04 的运行目录，必填
    --config：配置文件，默认读取运行目录里的 config.json
    --device：auto、cpu、cuda 或 mps，未写时用配置里的 device
    --base-checkpoint：完整微调模型或适配器目录，默认使用基础模型
    --episodes：仿真训练回合数，默认 4
    --max-steps：训练与评测的单回合步数上限，默认使用配置值
    --eval-episodes：每种策略的评测回合数，默认 2，须为偶数

被谁依赖：
    无
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p12_advance._impl.p12_cli import rl_main

if __name__ == "__main__":
    rl_main(__doc__)
