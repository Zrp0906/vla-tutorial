r"""在相同固定测试场景上比较专家、随机策略与未微调 SmolVLA。

输入：
    - 04_01 产生的 stats.json、03_01 产生的 manifest.json 测试清单。
    - 05_01 下载的真实 models/base/、models/processor/ 和 assets.json。

输出：
    - models/base/runtime.json 与 provenance.json；基础权重仍使用项目 models/base/，不再复制。
    - evaluations/expert/、random/、base/ 下的逐回合 JSON、按条件保存的 GIF 和 report.json。
    - 模型加载或闭环失败会停止并记录失败；产物描述不代表当前已经完成基线。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 需要独立运行，或交给 00_02_run_pipeline.py 按编号顺序调用。
    - 必须在 05_01 下载之后运行；随后分别开展部分参数训练和 LoRA。

运行参数：
    - `--run-dir`（必填；本次实验目录，本文以 /Users/zrp/Desktop/lora/smol/outputs/standard_run 为例；后续步骤复用同一目录）。
    - `--config`（可选 JSON 配置；默认 /Users/zrp/Desktop/lora/smol/configs/standard.json）。
    - `--device`（可选 auto/cpu/cuda/mps；未传时读取配置 device，标准配置为 auto）。
    - `--help`（显示实际参数并退出，不执行该阶段）。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p05_model_baseline/05_02_evaluate_baseline.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py`（入口路径引用/子进程调度）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p00_pipeline._impl.p00_cli import main
if __name__ == "__main__":
    main(5)
