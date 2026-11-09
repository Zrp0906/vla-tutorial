"""
功能：
    在 CUDA 上把指定 MLP 权重量化为 NF4，并用专家示范训练动作专家的 LoRA 适配器。

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
    CUDA 环境中的 bitsandbytes

输出：
    <run>/models/qlora/progress.json
    <run>/models/qlora/epoch_N/config.json
    <run>/models/qlora/epoch_N/runtime.json
    <run>/models/qlora/epoch_N/provenance.json
    <run>/models/qlora/epoch_N/adapter_model.safetensors
    <run>/models/qlora/epoch_N/adapter_config.json
    <run>/models/qlora/epoch_N/training_state.pt
    <run>/models/qlora/selected.json
    <run>/models/qlora/report.json

运行：
    独立运行。

参数：
    --run-dir：已完成阶段 03–04 的运行目录，必填
    --config：配置文件，默认读取运行目录里的 config.json
    --device：默认 cuda，本实验需要 NVIDIA CUDA
    --epochs：训练轮数，默认使用配置值
    --batch-size：每批窗口数，默认 1
    --window-stride：构造动作窗口的步长，默认使用配置值

被谁依赖：
    无
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p12_advance._impl.p12_cli import supervised_main
from smol_workflow.p12_advance._impl.p12_qlora import load_qlora

if __name__ == "__main__":
    supervised_main("qlora", load_qlora, __doc__)
