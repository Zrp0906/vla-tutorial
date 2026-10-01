r"""统一解析阶段参数、分派 01–11 实现并保存实际完成或失败状态。

输入：
    - 编号入口调用 main(number)；number 为阶段号 1–11。
    - 命令行配置、运行目录、设备及该阶段需要的上游数据、模型和报告。

输出：
    - 调用对应阶段实现；成功写 <run>/stages/NN.json，异常写错误与 traceback 后抛出。
    - 运行目录存在不同 config.json 时拒绝混用；本模块导入本身不执行阶段。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p01_environment/01_01_check_environment.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - `--run-dir`（必填；本次实验目录，本文以 /Users/zrp/Desktop/lora/smol/outputs/standard_run 为例；后续步骤复用同一目录）。
    - `--config`（可选 JSON 配置；默认 /Users/zrp/Desktop/lora/smol/configs/standard.json）。
    - `--device`（可选 auto/cpu/cuda/mps；未传时读取配置 device，标准配置为 auto）。
    - `--help`（显示实际参数并退出，不执行该阶段）。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p01_environment/01_01_check_environment.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p01_environment/01_01_check_environment.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p02_simulation/02_01_check_simulation.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p03_data_collection/03_01_collect_data.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p04_data_processing/04_01_process_data.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p05_model_baseline/05_02_evaluate_baseline.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/06_01_train_partial.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p07_lora_finetuning/07_01_train_lora.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p08_evaluation/08_01_evaluate_models.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p09_merge_quantization/09_01_merge_quantize.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/10_01_check_deployment.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p11_feedback_summary/11_01_summarize.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
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
