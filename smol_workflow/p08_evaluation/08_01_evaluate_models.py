"""
功能：
    在固定测试场景上对比部分参数模型和 LoRA 模型。

输入：
    configs/standard.json
    <run>/config.json（已存在时读取并核对）
    <run>/data/raw/manifest.json
    <run>/models/partial/selected.json
    <run>/models/lora/selected.json
    <run>/models/partial/epoch_N/runtime.json
    <run>/models/partial/epoch_N/model.safetensors
    <run>/models/lora/epoch_N/runtime.json
    <run>/models/lora/epoch_N/adapter_model.safetensors
    项目 models/base/config.json
    项目 models/base/model.safetensors
    项目 models/processor/ 下的处理器文件
    项目 models/assets.json

输出：
    <run>/evaluations/partial/episode_XXX.json
    <run>/evaluations/partial/episode_XXX.gif
    <run>/evaluations/partial/report.json
    <run>/evaluations/lora/episode_XXX.json
    <run>/evaluations/lora/episode_XXX.gif
    <run>/evaluations/lora/report.json
    <run>/stages/08.json

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
    main(8)
