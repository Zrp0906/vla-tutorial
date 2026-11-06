"""
功能：
    在固定测试场景上闭环评测专家、随机策略或指定模型。

输入：
    <run>/data/raw/manifest.json
    <run>/evaluations/<label>/episode_XXX.json（已存在时读取）

输出：
    <run>/evaluations/<label>/episode_XXX.json
    <run>/evaluations/<label>/episode_XXX.gif
    <run>/evaluations/<label>/report.json

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    p00_cli.py
"""
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
