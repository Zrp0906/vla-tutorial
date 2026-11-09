"""
功能：
    解析进阶入口的参数，复用阶段 03–04 的数据并分派训练。

输入：
    <run>/config.json 或 --config 指定的文件
    <run>/data/raw/manifest.json
    <run>/data/stats.json

输出：
    各实验的模型和报告由对应训练函数写入 <run>/models/。

运行：
    作为依赖导入，不单独运行。

参数：
    命令行参数由三个阶段入口传入。

被谁依赖：
    12_01_train_sft.py
    12_02_train_qlora.py
    12_03_train_residual_rl.py
"""
# 内部实现：由同阶段入口调用，不是独立执行步骤。
import argparse
from pathlib import Path

from smol_workflow.p00_pipeline._impl.p00_common import device, read_json


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("需要正整数")
    return number


def parser_for(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-dir", type=Path, required=True, help="已完成阶段 04 的目录")
    parser.add_argument("--config", type=Path, help="默认读取运行目录中的 config.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    return parser


def load_run(args):
    run = args.run_dir.resolve()
    for name in ("data/raw/manifest.json", "data/stats.json"):
        if not (run / name).is_file():
            raise FileNotFoundError(f"缺少 {run / name}，请先完成阶段 03–04")
    return run, read_json(args.config or run / "config.json")


def supervised_main(mode, initialize, description):
    parser = parser_for(description)
    parser.add_argument("--epochs", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--window-stride", type=positive_int)
    if mode == "sft":
        parser.add_argument("--text-layers", type=positive_int, default=1,
                            help="解冻多少个靠后的 VLM 文本层（跳过最后一层）")
    args = parser.parse_args()
    run, c = load_run(args)
    for key in ("epochs", "batch_size", "window_stride"):
        value = getattr(args, key)
        if value is not None:
            c[key] = value
    if mode == "sft":
        c["sft_text_layers"] = args.text_layers
    dev = device(args.device or ("cuda" if mode == "qlora" else c["device"]))

    from smol_workflow.p06_supervised_post_training._impl.p06_training import train

    train(c, run, dev, mode, initialize=initialize)
    print(f"{mode} completed: {run / 'models' / mode / 'report.json'}", flush=True)


def rl_main(description):
    parser = parser_for(description)
    parser.add_argument("--base-checkpoint", type=Path, help="可指定已有的模型或适配器目录")
    parser.add_argument("--episodes", type=positive_int, default=4, help="仿真训练回合数")
    parser.add_argument("--max-steps", type=positive_int, help="每回合步数上限")
    parser.add_argument("--eval-episodes", type=positive_int, default=2,
                        help="两种策略各评测多少局，填写偶数")
    args = parser.parse_args()
    if args.eval_episodes % 2:
        parser.error("--eval-episodes 需要是偶数，以覆盖两种指令")
    run, c = load_run(args)
    dev = device(args.device or c["device"])

    from smol_workflow.p12_advance._impl.p12_rl import train_residual

    train_residual(c, run, dev, args.base_checkpoint, args.episodes, args.max_steps, args.eval_episodes)
    print(f"residual_rl completed: {run / 'models' / 'residual_rl' / 'report.json'}", flush=True)
