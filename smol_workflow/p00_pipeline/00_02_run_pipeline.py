r"""按 01_01 到 11_01 的实际执行顺序串联 SmolVLA 全流程。

输入：
    - 统一配置、可选运行目录、设备和阶段起止范围；子进程复用当前 Python 解释器。
    - 下载固定在 05_01，紧接 05_02 基线；后续步骤读取前面步骤的真实产物。

输出：
    - 实验生成的 data/、models/、evaluations/、reports/ 和 stages/ 位于 --run-dir 内；05_01 的预训练资产另存项目 smol/models/。
    - --dry-run 只打印选中的执行顺序，不创建运行目录、不下载、不启动子进程；真实执行遇失败即停止。

独立运行：
    - 需要独立运行；这是按编号串联阶段脚本的总入口，不能让该入口递归调用自身。

运行参数：
    - `--config`（默认 /Users/zrp/Desktop/lora/smol/configs/standard.json）。
    - `--run-dir`（可选；默认 /Users/zrp/Desktop/lora/smol/outputs/run_<UTC时间>）。
    - `--device`（默认 auto，实际支持 auto/cpu/cuda/mps）。
    - `--from-stage`（整数 1–11，默认 1）；`--to-stage`（整数 1–11，默认 11）；按第一层阶段编号筛选。
    - `--dry-run`（开关，默认关闭；只列执行顺序）；`--help`（显示参数并退出）。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py --dry-run
    # 真正执行完整流程时去掉 --dry-run，可加 --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py`（动态导入、模拟调度与 --help 检查；不实际运行训练）。"""
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
