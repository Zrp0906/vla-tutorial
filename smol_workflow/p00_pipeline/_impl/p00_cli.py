"""
功能：
    解析阶段参数，分派阶段 01–11，并把完成或失败写到 stages/NN.json。

输入：
    configs/standard.json
    <run>/config.json（已存在时读取并核对）
    阶段 05 读取 <run>/data/stats.json
    阶段 05 读取 项目 models/assets.json
    阶段 05 读取 项目 models/base/model.safetensors

输出：
    <run>/stages/NN.json
    阶段 05 写入 <run>/models/base/runtime.json
    阶段 05 写入 <run>/models/base/provenance.json

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    01_01_check_environment.py
    02_01_check_simulation.py
    03_01_collect_data.py
    04_01_process_data.py
    05_02_evaluate_baseline.py
    06_01_train_partial.py
    07_01_train_lora.py
    08_01_evaluate_models.py
    09_01_merge_quantize.py
    10_01_check_deployment.py
    11_01_summarize.py
"""
import argparse
import traceback
from pathlib import Path
from smol_workflow.p00_pipeline._impl.p00_common import ROOT,config,device,write_json,read_json,now,sha256
from smol_workflow.p01_environment._impl.p01_environment import environment_report
from smol_workflow.p02_simulation._impl.p02_checks import simulation
from smol_workflow.p11_feedback_summary._impl.p11_summary import summary

def base_package(c,run):
    folder=run/"models"/"base"
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/"runtime.json",{"config":c,"stats":read_json(run/"data"/"stats.json"),
                                     "kind":"base","assets":read_json(ROOT/"models"/"assets.json")})
    write_json(folder/"provenance.json",{"kind":"base","weight_sha256":sha256(ROOT/"models"/"base"/"model.safetensors"),
                                        "weight_bytes":(ROOT/"models"/"base"/"model.safetensors").stat().st_size})
    return folder

def stage(number,c,run,dev):
    if number==1:
        return environment_report(c,run,dev)
    if number==2:
        return simulation(c,run)
    if number==3:
        from smol_workflow.p03_data_collection._impl.p03_data import collect
        return collect(c,run)
    if number==4:
        from smol_workflow.p04_data_processing._impl.p04_data import process
        return process(c,run)
    if number==5:
        from smol_workflow.p05_model_baseline._impl.p05_model import Runtime
        from smol_workflow.p08_evaluation._impl.p08_evaluation import evaluate
        result={name:evaluate(c,run,name) for name in ["expert","random"]}
        folder=run/"models"/"base"
        if not folder.exists():
            folder=base_package(c,run)
        runtime=Runtime(folder,dev)
        result["base"]=evaluate(c,run,"base",runtime)
        return result
    if number in (6,7):
        from smol_workflow.p06_supervised_post_training._impl.p06_training import train
        return train(c,run,dev,"partial" if number==6 else "lora")
    if number==8:
        from smol_workflow.p06_supervised_post_training._impl.p06_training import selected
        from smol_workflow.p05_model_baseline._impl.p05_model import Runtime
        from smol_workflow.p08_evaluation._impl.p08_evaluation import evaluate
        results={}
        for mode in ["partial","lora"]:
            runtime=Runtime(selected(run,mode),dev)
            results[mode]=evaluate(c,run,mode,runtime)
            del runtime
            import gc
            gc.collect()
        return results
    if number==9:
        from smol_workflow.p09_merge_quantization._impl.p09_quantization import convert
        from smol_workflow.p05_model_baseline._impl.p05_model import Runtime
        from smol_workflow.p08_evaluation._impl.p08_evaluation import evaluate
        result=convert(c,run,dev)
        for name in ["merged","int8"]:
            runtime=Runtime(run/"models"/name,dev)
            result[name]=evaluate(c,run,name,runtime)
            del runtime
            import gc
            gc.collect()
        return result
    if number==10:
        from smol_workflow.p10_deployment._impl.p10_deployment import check
        return check(c,run,dev,run/"models"/"merged")
    if number==11:
        return summary(c,run)
    raise ValueError(number)

def main(number):
    p=argparse.ArgumentParser(description=f"SmolVLA stage {number:02d}")
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--config",type=Path,default=ROOT/"configs"/"standard.json")
    p.add_argument("--device",default=None)
    a=p.parse_args()
    c=config(a.config)
    run=a.run_dir.resolve()
    run.mkdir(parents=True,exist_ok=True)
    if (run/"config.json").exists() and read_json(run/"config.json")!=c:
        raise ValueError("运行目录已有不同配置；请新建 run")
    dev=device(a.device or c["device"])
    status_path=run/"stages"/f"{number:02d}.json"
    started=now()
    try:
        stage(number,c,run,dev)
        write_json(status_path,{"stage":number,"status":"completed","started_at":started,"finished_at":now()})
        print(f"STAGE {number:02d} completed",flush=True)
    except BaseException as exc:
        write_json(status_path,{"stage":number,"status":"failed","started_at":started,
                               "error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc()})
        raise
