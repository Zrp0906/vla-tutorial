"""
功能：
    汇总训练、评测、量化和部署结果，并收集失败场景。

输入：
    configs/standard.json
    <run>/config.json（已存在时读取并核对）
    <run>/evaluations/*/report.json
    <run>/evaluations/base/episode_XXX.json
    <run>/evaluations/partial/episode_XXX.json
    <run>/evaluations/lora/episode_XXX.json
    <run>/evaluations/merged/episode_XXX.json
    <run>/evaluations/int8/episode_XXX.json
    <run>/models/partial/report.json
    <run>/models/lora/report.json
    <run>/reports/quantization.json
    <run>/reports/deployment.json

输出：
    <run>/reports/summary.json
    <run>/reports/failure_cases.json
    <run>/reports/REPORT.md
    <run>/stages/11.json

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
    main(11)
