r"""采集两条语言指令的成功示范并保存轨迹与独立测试清单。

输入：
    - 统一配置、运行目录和脚本专家；默认 50 条成功示范，每条指令 25 条。
    - 按完整轨迹划分 40 条训练、10 条验证；另保存 20 个固定独立测试场景。

输出：
    - data/raw/episode_XXX/ 下的逐帧 PNG 与 trajectory.json。
    - data/raw/manifest.json 记录轨迹划分、失败尝试与测试 seed/goal；reports/expert_goal_0.gif、expert_goal_1.gif。
    - 原始数据目录或清单已存在时拒绝覆盖，使用新运行目录。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 需要独立运行，或交给 00_02_run_pipeline.py 按编号顺序调用。
    - 先检查仿真，再采集；完成后执行 04_01 数据处理。

运行参数：
    - `--run-dir`（必填；本次实验目录，本文以 /Users/zrp/Desktop/lora/smol/outputs/standard_run 为例；后续步骤复用同一目录）。
    - `--config`（可选 JSON 配置；默认 /Users/zrp/Desktop/lora/smol/configs/standard.json）。
    - `--device`（可选 auto/cpu/cuda/mps；未传时读取配置 device，标准配置为 auto）。
    - `--help`（显示实际参数并退出，不执行该阶段）。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p03_data_collection/03_01_collect_data.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py`（入口路径引用/子进程调度）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p00_pipeline._impl.p00_cli import main
if __name__ == "__main__":
    main(3)
