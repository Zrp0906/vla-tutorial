r"""冻结视觉语言骨干，对动作专家和相关投影进行两轮监督后训练。

输入：
    - 真实基础模型、训练/验证原始轨迹和仅训练集的归一化统计量。
    - 标准配置 epochs=2、batch_size=2、partial_lr=1e-4；Flow Matching 仅归约有效时间步与二维动作。

输出：
    - models/partial/epoch_1/、epoch_2/ 的推理包与 training_state.pt。
    - models/partial/progress.json、selected.json 与 report.json；按验证损失选择 checkpoint。
    - 校验冻结参数不变、可训练参数更新；本入口没有 --resume 参数，保存状态不等于已实现恢复入口。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 需要独立运行，或交给 00_02_run_pipeline.py 按编号顺序调用。
    - 在数据和模型准备完成后执行；不把两轮训练当作策略效果保证。

运行参数：
    - `--run-dir`（必填；本次实验目录，本文以 /Users/zrp/Desktop/lora/smol/outputs/standard_run 为例；后续步骤复用同一目录）。
    - `--config`（可选 JSON 配置；默认 /Users/zrp/Desktop/lora/smol/configs/standard.json）。
    - `--device`（可选 auto/cpu/cuda/mps；未传时读取配置 device，标准配置为 auto）。
    - `--help`（显示实际参数并退出，不执行该阶段）。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/06_01_train_partial.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py`（入口路径引用/子进程调度）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p00_pipeline._impl.p00_cli import main
if __name__ == "__main__":
    main(6)
