"""
功能：
    读写配置和 JSON，选择设备，固定随机种子，计算哈希，保存 GIF。

输入：
    configs/standard.json
    调用方给出的 JSON 文件
    调用方给出的二进制文件

输出：
    调用方给出的 JSON 文件
    调用方给出的 GIF 文件

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    00_01_check_structure.py
    p00_cli.py
    p01_environment.py
    p02_checks.py
    p03_data.py
    p04_data.py
    p05_model.py
    p06_training.py
    p08_evaluation.py
    p09_quantization.py
    p10_deployment.py
    p10_server.py
    p11_summary.py
"""
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
    import torch
    system = platform.system()
    if system == "Windows":
        import psutil

        memory = psutil.Process().memory_info()
        peak = getattr(memory, "peak_rss", None)
        if peak is None:
            peak = memory.peak_wset
    else:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(rss if system == "Darwin" else rss * 1024)
    result = {"process_peak_rss_bytes": peak}
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
