r"""验证二维推块仿真中的专家控制与实际 RGB 回放。

输入：
    - 统一仿真配置与运行目录；使用固定 seed 分别检查蓝色、绿色目标。

输出：
    - reports/simulation_goal_0.gif、simulation_goal_1.gif，图像来自真实环境状态。
    - 专家结果返回给调度器；任一任务失败即抛出异常。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 需要独立运行，或交给 00_02_run_pipeline.py 按编号顺序调用。
    - 在环境检查之后、数据采集之前执行。

运行参数：
    - `--run-dir`（必填；本次实验目录，本文以 /Users/zrp/Desktop/lora/smol/outputs/standard_run 为例；后续步骤复用同一目录）。
    - `--config`（可选 JSON 配置；默认 /Users/zrp/Desktop/lora/smol/configs/standard.json）。
    - `--device`（可选 auto/cpu/cuda/mps；未传时读取配置 device，标准配置为 auto）。
    - `--help`（显示实际参数并退出，不执行该阶段）。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p02_simulation/02_01_check_simulation.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py`（入口路径引用/子进程调度）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p00_pipeline._impl.p00_cli import main
if __name__ == "__main__":
    main(2)
