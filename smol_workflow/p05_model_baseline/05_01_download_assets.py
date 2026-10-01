r"""获取真实 SmolVLA 权重与处理器文件，并记录上游不可变 revision。

输入：
    - 网络可访问 Hugging Face；基础仓库 lerobot/smolvla_base 与处理器仓库 HuggingFaceTB/SmolVLM2-500M-Video-Instruct。
    - 若 models/assets.json 已存在，只打印现有清单并返回；未存在时解析仓库当前 SHA 后按该 revision 下载。

输出：
    - 项目 models/base/、models/processor/ 和 models/assets.json，不写入实验 --run-dir。
    - 只下载处理器定义与分词器文件，不再下载第二份完整 VLM；模型缓存使用项目 HF_HOME。
    - 不安装或替换 embodiedai；不上传 Hub，也不在此脚本验证前向/反向。

独立运行：
    - 需要独立运行，或交给 00_02_run_pipeline.py 按编号顺序调用。

运行参数：
    - `--help`（显示说明并退出）；没有 --config、--run-dir 或 --device 参数。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p05_model_baseline/05_01_download_assets.py
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py`（入口路径引用/子进程调度）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
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
