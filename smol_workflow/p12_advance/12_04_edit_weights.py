"""
功能：
    诊断同义指令故障，评测大模型提出的 SmolVLA 权重补丁。

输入：
    --run-dir 中的配置、测试场景与归一化统计量
    项目基础模型及处理器，或已有的浮点推理包
    可选的大模型 JSON 补丁

输出：
    实验目录中的 diagnosis.json、experiment.json 和诊断回放
    有补丁时另写 comparison/patch.json、report.json 和对照回放

运行：
    独立运行。先诊断，再使用 --patch 进行对照评测。

参数：
    --run-dir：已完成阶段 03–04 的运行目录，必填
    --config、--device：配置和设备
    --experiment-dir：实验目录，默认使用运行目录下的 models/wsp
    --base-checkpoint：诊断使用的浮点推理包，默认使用基础模型
    --seeds-per-split：诊断集和留出集各取多少个种子，默认 2
    --max-steps：诊断和后续评测共用的回合步数上限
    --patch：已准备好的补丁文件，省略时执行诊断

被谁依赖：
    无
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p12_advance._impl.p12_cli import wsp_main

if __name__ == "__main__":
    wsp_main(__doc__)
