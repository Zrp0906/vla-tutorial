"""
功能：
    冻结视觉语言骨干，做部分参数监督训练。

输入：
    configs/standard.json
    <run>/config.json（已存在时读取并核对）
    <run>/data/raw/manifest.json
    <run>/data/raw/episode_XXX/trajectory.json
    <run>/data/raw/episode_XXX/帧号.png
    <run>/data/stats.json
    项目 models/base/config.json
    项目 models/base/model.safetensors
    项目 models/processor/ 下的处理器文件
    项目 models/assets.json

输出：
    <run>/models/partial/progress.json
    <run>/models/partial/epoch_N/config.json
    <run>/models/partial/epoch_N/runtime.json
    <run>/models/partial/epoch_N/provenance.json
    <run>/models/partial/epoch_N/model.safetensors
    <run>/models/partial/epoch_N/training_state.pt
    <run>/models/partial/selected.json
    <run>/models/partial/report.json
    <run>/stages/06.json

运行：
    可独立运行，也可由 00_02_run_pipeline.py 调用。

参数：
    --run-dir：实验目录，必填
    --config：配置文件，默认 configs/standard.json
    --device：auto、cpu、cuda 或 mps，未写时用配置里的 device

被谁依赖：
    00_02_run_pipeline.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p00_pipeline._impl.p00_cli import main
if __name__ == "__main__":
    main(6)
