r"""从同一基础模型独立训练 LoRA，并保存适配器与参数更新检查结果。

输入：
    - 与 06_01 相同的基础模型、训练/验证轨迹和统计量，不接着 partial checkpoint 训练。
    - 标准配置 epochs=2、lora_rank=8、lora_alpha=16、lora_lr=1e-4。

输出：
    - models/lora/epoch_1/、epoch_2/ 的 adapter_config.json、adapter_model.safetensors、配置、来源与 training_state.pt。
    - models/lora/progress.json、selected.json 和 report.json；统计可训练参数并验收冻结/更新。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 需要独立运行，或交给 00_02_run_pipeline.py 按编号顺序调用。
    - 按总流程在 06_01 后运行，但两个实验各自从基础模型开始，复用同一训练循环。

运行参数：
    - `--run-dir`（必填；本次实验目录，本文以 /Users/zrp/Desktop/lora/smol/outputs/standard_run 为例；后续步骤复用同一目录）。
    - `--config`（可选 JSON 配置；默认 /Users/zrp/Desktop/lora/smol/configs/standard.json）。
    - `--device`（可选 auto/cpu/cuda/mps；未传时读取配置 device，标准配置为 auto）。
    - `--help`（显示实际参数并退出，不执行该阶段）。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p07_lora_finetuning/07_01_train_lora.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py`（入口路径引用/子进程调度）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p00_pipeline._impl.p00_cli import main
if __name__ == "__main__":
    main(7)
