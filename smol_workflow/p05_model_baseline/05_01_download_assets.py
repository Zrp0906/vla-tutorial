"""
功能：
    下载 SmolVLA 权重和处理器，已有 models/assets.json 时只打印清单。

输入：
    项目 models/assets.json（已存在时只读取）

输出：
    项目 models/base/config.json
    项目 models/base/model.safetensors
    项目 models/processor/ 下的处理器文件
    项目 models/assets.json

运行：
    可独立运行，也可由 00_02_run_pipeline.py 调用。

参数：
    无

被谁依赖：
    00_02_run_pipeline.py
"""
# 执行步骤 05_01：在数据处理之后、基线评测之前获取模型资产。
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
from huggingface_hub import HfApi, snapshot_download

def main():
    manifest = ROOT / "models" / "assets.json"
    if manifest.exists():
        print(manifest.read_text(encoding="utf-8"), flush=True)
        return
    assets = {}
    for name, repo, patterns in [
        ("base", "lerobot/smolvla_base", ["*.json", "*.safetensors"]),
        ("processor", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", ["*.json", "*.txt", "*.model", "*.jinja"]),
    ]:
        revision = HfApi().model_info(repo).sha
        destination = ROOT / "models" / name
        print(f"Downloading {repo}@{revision} -> {destination}", flush=True)
        snapshot_download(repo, revision=revision, local_dir=destination, allow_patterns=patterns, max_workers=2)
        assets[name] = {"repo_id": repo, "revision": revision, "path": str(destination.relative_to(ROOT))}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(assets, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(assets, indent=2), flush=True)

if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
