r"""为专家、随机和真实模型共用固定测试清单的闭环评测器。

输入：
    - evaluate(c, run, label, runtime=None)；data/raw/manifest.json 中的 tests 列表。
    - runtime 存在时每次仅执行动作块第 1 步；否则 label=expert 使用专家，其余无 runtime 的调用使用随机控制。

输出：
    - evaluations/<label>/episode_XXX.json、条件性 GIF 与 report.json。
    - 记录动作、距离、成功率、步数、延迟和异常；已存在回合 JSON 会被复用，出现新异常先记录再抛出。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p08_evaluation/08_01_evaluate_models.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本模块无命令行；label 指定结果子目录，runtime 指定真实模型推理接口。
    - 阶段 05、08、09 共用该评测器；下方 08_01 命令要求两项训练的 selected.json 已存在。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p08_evaluation/08_01_evaluate_models.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
import time
import numpy as np
from smol_workflow.p00_pipeline._impl.p00_common import read_json,write_json,save_gif,memory_report,sha256,now
from smol_workflow.p02_simulation._impl.p02_environment import make_env,Expert

def evaluate(c,run,label,runtime=None):
    tests=read_json(run/"data"/"raw"/"manifest.json")["tests"]
    folder=run/"evaluations"/label
    folder.mkdir(parents=True,exist_ok=True)
    episodes=[]
    for index,case in enumerate(tests):
        path=folder/f"episode_{index:03d}.json"
        if path.exists():
            episodes.append(read_json(path))
            continue
        env=make_env(c)
        obs,_=env.reset(seed=case["seed"],options={"goal":case["goal"]})
        expert=Expert(env) if label=="expert" else None
        if runtime:
            runtime.reset()
        frames=[obs["image"]]
        actions,latencies=[],[]
        started=time.perf_counter()
        error=None
        try:
            while True:
                if runtime:
                    chunk,seconds=runtime.predict(obs,c["eval_noise_seed"]+index*1000+env.steps)
                    action=chunk[0]
                    latencies.append(seconds)
                elif expert:
                    action=expert.action()
                else:
                    action=np.random.default_rng(case["seed"]+env.steps).uniform(-1,1,2).astype(np.float32)
                actions.append(np.asarray(action).tolist())
                obs,_,done,timeout,info=env.step(action)
                frames.append(obs["image"])
                if done or timeout:
                    break
        except Exception as exc:
            error=f"{type(exc).__name__}: {exc}"
            info=env.info()
        item={"index":index,**case,**info,"completed":error is None,"error":error,
              "seconds":time.perf_counter()-started,"actions":actions,
              "latency_mean_seconds":float(np.mean(latencies)) if latencies else None,
              "latency_p95_seconds":float(np.percentile(latencies,95)) if latencies else None,
              "recorded_at":now()}
        write_json(path,item)
        if index<2 or item["success"]:
            save_gif(folder/f"episode_{index:03d}.gif",frames,c["fps"])
        episodes.append(item)
        print(f"{label}: episode {index+1}/{len(tests)} success={item['success']} steps={item['steps']} seconds={item['seconds']:.1f}",flush=True)
        if error:
            raise RuntimeError(error)
    result={"label":label,"profile":c["profile"],"config":c,
            "test_manifest_sha256":sha256(run/"data"/"raw"/"manifest.json"),
            "successes":sum(e["success"] for e in episodes),"episodes":len(episodes),
            "success_rate":sum(e["success"] for e in episodes)/len(episodes),
            "mean_final_distance":float(np.mean([e["distance"] for e in episodes])),
            "mean_steps":float(np.mean([e["steps"] for e in episodes])),
            "mean_inference_latency_seconds":float(np.mean([e["latency_mean_seconds"] for e in episodes])) if runtime else None,
            "errors":sum(e["error"] is not None for e in episodes),"recorded_at":now()}
    if runtime:
        result.update(device=str(runtime.dev),**memory_report(runtime.dev))
    write_json(folder/"report.json",result)
    return result
