r"""在新进程启动本机 HTTP 推理并核对直接调用、异常输入和闭环控制。

输入：
    - models/merged/ 真实推理包、统一配置和基础处理器资产。
    - 同一观测与噪声种子 123 用于直接/HTTP 输出对比；服务仅绑定 127.0.0.1。

输出：
    - reports/deployment.json、server.log、http_goal_0.gif 与 http_goal_1.gif。
    - 检查新进程重载、输出等价、非法 state 返回 HTTP 400，并运行两条指令的闭环。
    - 无论检查成功或失败，finally 都终止该入口启动的服务子进程。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 需要独立运行，或交给 00_02_run_pipeline.py 按编号顺序调用。
    - 先合并/量化，再做部署验收；服务由检查程序自动启动，无需手动先开服务器。

运行参数：
    - `--run-dir`（必填；本次实验目录，本文以 /Users/zrp/Desktop/lora/smol/outputs/standard_run 为例；后续步骤复用同一目录）。
    - `--config`（可选 JSON 配置；默认 /Users/zrp/Desktop/lora/smol/configs/standard.json）。
    - `--device`（可选 auto/cpu/cuda/mps；未传时读取配置 device，标准配置为 auto）。
    - `--help`（显示实际参数并退出，不执行该阶段）。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/10_01_check_deployment.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py`（入口路径引用/子进程调度）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p00_pipeline._impl.p00_cli import main
if __name__ == "__main__":
    main(10)
