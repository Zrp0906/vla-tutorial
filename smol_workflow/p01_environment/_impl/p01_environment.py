"""
功能：
    记录 Python、设备、关键包版本和 pip 检查结果。

输入：
    无

输出：
    <run>/config.json
    <run>/reports/environment.json
    <run>/reports/requirements.lock.txt
    <run>/reports/pip_check.txt

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    p00_cli.py
"""
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
