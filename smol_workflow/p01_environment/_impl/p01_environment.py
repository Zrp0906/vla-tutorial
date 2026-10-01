r"""检查已有 embodiedai 的设备、依赖一致性并记录环境快照。

输入：
    - 统一配置、运行目录和当前 Python 解释器；关键包已安装。
    - 读取 CUDA/MPS 可用性，并在选定设备实际创建张量；不会安装或升级依赖。

输出：
    - <run>/config.json 与 reports/environment.json。
    - reports/requirements.lock.txt、reports/pip_check.txt；依赖冲突或张量创建失败时停止。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p01_environment/01_01_check_environment.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本文件不解析命令行；environment_report(c, run, dev) 接收配置、运行路径和实际 torch.device。
    - 下方环境入口的 --run-dir 必填，--config 与 --device 可选，详见该入口说明。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p01_environment/01_01_check_environment.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
import importlib.metadata
import platform
import subprocess
import sys
from smol_workflow.p00_pipeline._impl.p00_common import write_json

def environment_report(c,run,dev):
    import torch
    report={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
            "device":str(dev),"mps_built":torch.backends.mps.is_built(),
            "mps_available":torch.backends.mps.is_available(),"cuda_available":torch.cuda.is_available(),
            "packages":{p:importlib.metadata.version(p) for p in ["torch","lerobot","transformers","peft","numpy","Pillow","gymnasium"]},
            "packages_before_change":"no environment packages modified"}
    run.mkdir(parents=True,exist_ok=True)
    write_json(run/"config.json",c)
    write_json(run/"reports"/"environment.json",report)
    frozen=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,check=True).stdout
    (run/"reports"/"requirements.lock.txt").write_text(frozen,encoding="utf-8")
    check=subprocess.run([sys.executable,"-m","pip","check"],capture_output=True,text=True)
    (run/"reports"/"pip_check.txt").write_text(check.stdout+check.stderr,encoding="utf-8")
    if check.returncode:
        raise RuntimeError("依赖不一致，查看 pip_check.txt")
    assert float(torch.ones(1,device=dev).cpu()[0])==1
    return report
