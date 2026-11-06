"""
功能：
    从基础模型单独做 LoRA 微调。

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
    <run>/models/lora/progress.json
    <run>/models/lora/epoch_N/config.json
    <run>/models/lora/epoch_N/runtime.json
    <run>/models/lora/epoch_N/provenance.json
    <run>/models/lora/epoch_N/adapter_config.json
    <run>/models/lora/epoch_N/training_state.pt
    <run>/models/lora/selected.json
    <run>/models/lora/report.json
    <run>/models/lora/epoch_N/adapter_model.safetensors
    <run>/stages/07.json

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
    main(7)
