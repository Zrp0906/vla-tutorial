"""
功能：
    检查二维推块仿真和专家控制，并保存回放。

输入：
    configs/standard.json
    <run>/config.json（已存在时读取并核对）

输出：
    <run>/reports/simulation_goal_0.gif
    <run>/reports/simulation_goal_1.gif
    <run>/stages/02.json

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
    main(2)
