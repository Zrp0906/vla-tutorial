r"""合并 LoRA，校验固定噪声输出，再生成教学 INT8 权重存储版本。

输入：
    - 选中的 LoRA 适配器、对应基础模型、归一化统计量与固定测试场景。
    - 只量化 lm_expert 中的 Linear 权重；输入输出投影等未选中模块保留浮点。

输出：
    - models/merged/ 完整浮点推理包；models/int8/ 含 INT8 权重、逐通道 scale 和 quantization.json。
    - reports/quantization.json 保存合并/量化差异、覆盖率与文件大小；evaluations/merged/、int8/ 保存闭环结果。
    - 采用按层反量化后浮点计算，不宣称 INT8 内核加速；合并数值对比不通过则停止。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 需要独立运行，或交给 00_02_run_pipeline.py 按编号顺序调用。
    - 依赖 LoRA 训练；成功产物供 10_01 部署检查使用。

运行参数：
    - `--run-dir`（必填；本次实验目录，本文以 /Users/zrp/Desktop/lora/smol/outputs/standard_run 为例；后续步骤复用同一目录）。
    - `--config`（可选 JSON 配置；默认 /Users/zrp/Desktop/lora/smol/configs/standard.json）。
    - `--device`（可选 auto/cpu/cuda/mps；未传时读取配置 device，标准配置为 auto）。
    - `--help`（显示实际参数并退出，不执行该阶段）。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p09_merge_quantization/09_01_merge_quantize.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_02_run_pipeline.py`（入口路径引用/子进程调度）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from smol_workflow.p00_pipeline._impl.p00_cli import main
if __name__ == "__main__":
    main(9)
