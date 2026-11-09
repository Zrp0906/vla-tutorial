"""
功能：
    检查阶段目录、入口编号、内部模块导入和总入口调度顺序，不下载、不训练。

输入：
    smol_workflow/**/*.py
    configs/standard.json

输出：
    无

运行：
    独立运行。

参数：
    无

被谁依赖：
    无
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    package = ROOT / "smol_workflow"
    stages = sorted(
        p for p in package.iterdir()
        if p.is_dir() and re.match(r"p\d{2}_", p.name) and any(p.rglob("*.py"))
    )
    assert [int(p.name[1:3]) for p in stages] == list(range(13)), "缺少或重复的阶段目录"
    sources = sorted(package.rglob("*.py"))
    imported, entries = 0, []
    for source in sources:
        relative = source.relative_to(package)
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        if source.name == "__init__.py":
            continue
        stage = relative.parts[0][1:3]
        if "_impl" in relative.parts:
            assert len(relative.parts) == 3 and source.stem.startswith("p" + stage + "_")
            name = ".".join(("smol_workflow", *relative.with_suffix("").parts))
            importlib.import_module(name)
            imported += 1
        else:
            assert len(relative.parts) == 2, f"入口不在阶段目录：{relative}"
            assert re.match(rf"{stage}_\d{{2}}_", source.name), f"入口缺少执行序号：{relative}"
            entries.append(source)

    numbers = [tuple(map(int, p.stem.split("_")[:2])) for p in entries]
    assert len(numbers) == len(set(numbers)), "执行入口编号重复"
    for stage in range(13):
        substeps = sorted(second for first, second in numbers if first == stage)
        assert substeps == list(range(1, len(substeps) + 1)), f"阶段 {stage:02d} 步骤不连续"

    from smol_workflow.p00_pipeline._impl.p00_common import ROOT as runtime_root, config
    assert runtime_root == ROOT, "内部模块的项目根目录解析错误"
    assert config()["profile"] == "standard", "默认配置应为原计划的 standard"
    pipeline = importlib.import_module("smol_workflow.p00_pipeline.00_02_run_pipeline")
    expected = [p for p in entries if 1 <= int(p.name[:2]) <= 11]
    scheduled = [ROOT / script for _, script, _ in pipeline.STEPS]
    assert scheduled == expected, "总入口执行顺序与文件编号不一致"
    for stage, script, _ in pipeline.STEPS:
        assert stage == int(Path(script).name.split("_")[0]), "调度阶段与文件编号不一致"

    # 不仅检查列表，还模拟总入口，验证实际子进程调用的顺序及失败即停。
    with tempfile.TemporaryDirectory(prefix="smol-order-check-") as temporary:
        run = Path(temporary) / "run"
        args = ["--run-dir", str(run), "--device", "cpu"]
        with patch.object(pipeline.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(args + ["--dry-run"])
            assert not run.exists() and not child.called, "dry-run 产生了执行副作用"
            pipeline.main(args)
            assert [Path(call.args[0][1]) for call in child.call_args_list] == expected
        with patch.object(pipeline.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(args + ["--from-stage", "5", "--to-stage", "5"])
            assert [Path(call.args[0][1]).name for call in child.call_args_list] == [
                "05_01_download_assets.py", "05_02_evaluate_baseline.py"]
        with patch.object(pipeline.subprocess, "run",
                          side_effect=[None, subprocess.CalledProcessError(1, "simulated")]) as child:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.main(args)
            except subprocess.CalledProcessError:
                assert child.call_count == 2
            else:
                raise AssertionError("上游失败后没有停止")

    for entry in entries:
        result = subprocess.run([sys.executable, "-B", str(entry), "--help"],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f"入口检查失败：{entry}\n{result.stdout}\n{result.stderr}")
        assert "usage:" in result.stdout

    result = subprocess.run(
        [sys.executable, "-B", "-m", "smol_workflow.p10_deployment._impl.p10_server", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"HTTP 子进程入口失败：{result.stderr}")
    assert "--checkpoint" in result.stdout
    print(f"PASS: {len(stages)} stages, {len(sources)} Python files, {imported} implementation imports, "
          f"{len(entries)} uniquely ordered CLI entries and HTTP --help; "
          "dispatch order, stage 05 order, dry-run and fail-fast verified; "
          "no training or download executed.")


if __name__ == "__main__":
    main()
