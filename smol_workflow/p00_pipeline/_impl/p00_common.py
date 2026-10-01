r"""提供 SmolVLA 全流程的配置、设备、序列化、随机种子与资源记录工具。

输入：
    - 函数传入的 JSON 路径、数据、seed、torch.device 或 RGB 帧。
    - 默认配置 configs/standard.json；ROOT 指向 smol，导入时设置项目 HF_HOME 与禁用 Xet 的默认环境变量。

输出：
    - 返回配置、设备、哈希、时间或资源记录；write_json 原子替换目标 JSON，save_gif 写实际帧回放。
    - memory_report 使用 Unix resource 模块，Windows 的该函数尚非实测兼容；导入本模块不启动训练。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p01_environment/01_01_check_environment.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本文件不解析命令行；常用函数为 config(path=None)、device(request='auto')、write_json(path, value)、save_gif(path, frames, fps=10)。
    - 下方用法通过 01_01 环境入口调用公共工具；该入口参数见对应模块文档。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p01_environment/01_01_check_environment.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p01_environment/_impl/p01_environment.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p02_simulation/_impl/p02_checks.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p03_data_collection/_impl/p03_data.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p04_data_processing/_impl/p04_data.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p05_model_baseline/_impl/p05_model.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/_impl/p06_training.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p08_evaluation/_impl/p08_evaluation.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p09_merge_quantization/_impl/p09_quantization.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/_impl/p10_deployment.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/_impl/p10_server.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p11_feedback_summary/_impl/p11_summary.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
# 按工作流阶段组织，入口与实现同目录。
from __future__ import annotations
import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)

def config(path=None):
    c = read_json(path or ROOT / "configs" / "standard.json")
    if c["epochs"] != 2 or c["train_episodes"] >= c["episodes"]:
        raise ValueError("首轮必须为 2 epoch 且包含独立验证轨迹")
    if c["episodes"] % 2 or c["train_episodes"] % 2 or c["test_episodes"] % 2:
        raise ValueError("两条指令要求均衡的回合数量")
    return c

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def device(request="auto"):
    import torch
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if request == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前进程 MPS 不可见；请在普通终端或获准的沙箱外运行，不要据此判定硬件不支持 MPS")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前进程没有可用 CUDA")
    if request not in {"cpu", "cuda", "mps"}:
        raise ValueError(request)
    return torch.device(request)

def synchronize(dev):
    import torch
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()

def memory_report(dev):
    import resource
    import torch
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_bytes": int(rss if platform.system() == "Darwin" else rss * 1024)}
    if dev.type == "mps":
        result.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
    elif dev.type == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return result

def now():
    return datetime.now(timezone.utc).isoformat()

def save_gif(path, frames, fps=10):
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000//fps, loop=0)
