"""
功能：
    按 01–11 的顺序用子进程跑完全流程，某一步失败就停止。

输入：
    无

输出：
    无

运行：
    可独立运行，也可被 00_01_check_structure.py 导入。

参数：
    --config：配置文件，默认 configs/standard.json
    --run-dir：实验目录，未写时用 outputs/run_<UTC时间>
    --device：auto、cpu、cuda 或 mps，默认 auto
    --from-stage：起始阶段，1–11，默认 1
    --to-stage：结束阶段，1–11，默认 11
    --dry-run：只打印执行顺序，不创建目录、不跑步骤

被谁依赖：
    00_01_check_structure.py
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# (阶段、入口、是否需要统一的运行参数)，排列即真实子进程执行顺序。
STEPS = [
    (1, "smol_workflow/p01_environment/01_01_check_environment.py", True),
    (2, "smol_workflow/p02_simulation/02_01_check_simulation.py", True),
    (3, "smol_workflow/p03_data_collection/03_01_collect_data.py", True),
    (4, "smol_workflow/p04_data_processing/04_01_process_data.py", True),
    (5, "smol_workflow/p05_model_baseline/05_01_download_assets.py", False),
    (5, "smol_workflow/p05_model_baseline/05_02_evaluate_baseline.py", True),
    (6, "smol_workflow/p06_supervised_post_training/06_01_train_partial.py", True),
    (7, "smol_workflow/p07_lora_finetuning/07_01_train_lora.py", True),
    (8, "smol_workflow/p08_evaluation/08_01_evaluate_models.py", True),
    (9, "smol_workflow/p09_merge_quantization/09_01_merge_quantize.py", True),
    (10, "smol_workflow/p10_deployment/10_01_check_deployment.py", True),
    (11, "smol_workflow/p11_feedback_summary/11_01_summarize.py", True),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="按文件编号执行 SmolVLA 流程")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "standard.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--from-stage", type=int, default=1, choices=range(1, 12))
    parser.add_argument("--to-stage", type=int, default=11, choices=range(1, 12))
    parser.add_argument("--dry-run", action="store_true", help="只列出真实执行顺序，不创建运行目录、不执行步骤")
    args = parser.parse_args(argv)
    if args.from_stage > args.to_stage:
        parser.error("--from-stage 不能大于 --to-stage")
    run = (args.run_dir or ROOT / "outputs" / (
        "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))).resolve()
    commands = []
    for stage, script, needs_run_args in STEPS:
        if args.from_stage <= stage <= args.to_stage:
            path = ROOT / script
            if not path.is_file():
                raise FileNotFoundError(path)
            command = [sys.executable, str(path)]
            if needs_run_args:
                command.extend(["--config", str(args.config.resolve()),
                                "--run-dir", str(run), "--device", args.device])
            commands.append((path, command))

    if args.dry_run:
        print("Execution order (no steps executed):")
        for path, _ in commands:
            print(path.relative_to(ROOT))
        return

    run.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    print(f"RUN_DIR={run}", flush=True)
    for path, command in commands:
        print(f"Starting {path.name}", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print(f"Completed requested steps. Reports: {run / 'reports'}", flush=True)


if __name__ == "__main__":
    main()
