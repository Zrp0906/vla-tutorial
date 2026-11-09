"""
功能：
    用专家示范训练动作专家、相关投影和指定数量的 VLM 文本层。

输入：
    <run>/config.json（默认读取，也可用 --config 指定原始配置）
    <run>/data/raw/manifest.json
    <run>/data/raw/episode_XXX/trajectory.json
    <run>/data/raw/episode_XXX/帧号.png
    <run>/data/stats.json
    项目 models/base/config.json
    项目 models/base/model.safetensors
    项目 models/processor/ 下的处理器文件
    项目 models/assets.json

输出：
    <run>/models/sft/progress.json
    <run>/models/sft/epoch_N/config.json
    <run>/models/sft/epoch_N/runtime.json
    <run>/models/sft/epoch_N/provenance.json
    <run>/models/sft/epoch_N/model.safetensors
    <run>/models/sft/epoch_N/training_state.pt
    <run>/models/sft/selected.json
    <run>/models/sft/report.json

运行：
    独立运行。

参数：
    --run-dir：已完成阶段 03–04 的运行目录，必填
    --config：配置文件，默认读取运行目录里的 config.json
    --device：auto、cpu、cuda 或 mps，未写时用配置里的 device
    --epochs：训练轮数，默认使用配置值
    --batch-size：每批窗口数，默认 1
    --window-stride：构造动作窗口的步长，默认使用配置值
    --text-layers：解冻的靠后文本层数，默认 1，跳过最后一层

被谁依赖：
    无
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p12_advance._impl.p12_cli import supervised_main
from smol_workflow.p12_advance._impl.p12_sft import load_sft

if __name__ == "__main__":
    supervised_main("sft", load_sft, __doc__)
