"""
功能：
    检查新进程重载、本机 HTTP 推理和仿真闭环。

输入：
    configs/standard.json
    <run>/config.json（已存在时读取并核对）
    <run>/models/merged/config.json
    <run>/models/merged/runtime.json
    <run>/models/merged/model.safetensors
    项目 models/processor/ 下的处理器文件

输出：
    <run>/reports/server.log
    <run>/reports/deployment.json
    <run>/reports/http_goal_0.gif
    <run>/reports/http_goal_1.gif
    <run>/stages/10.json

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
    main(10)
