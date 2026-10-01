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
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/10_01_check_deployment.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本模块无命令行；check(c, run, dev, checkpoint) 自动选择空闲本机端口并创建/终止服务器子进程。
    - request(url, payload=None) 的请求超时与服务启动等待均为 180 秒；用 10_01 入口执行检查。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/10_01_check_deployment.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
import base64
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import numpy as np
from PIL import Image
from smol_workflow.p00_pipeline._impl.p00_common import ROOT,write_json,save_gif
from smol_workflow.p02_simulation._impl.p02_environment import make_env
from smol_workflow.p05_model_baseline._impl.p05_model import Runtime

def request(url,payload=None):
    data=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=180) as response:
        return json.load(response)

def payload(obs,seed):
    memory=io.BytesIO()
    Image.fromarray(obs["image"]).save(memory,format="PNG")
    return {"state":obs["state"].tolist(),"task":obs["task"],"noise_seed":seed,
            "image_png_base64":base64.b64encode(memory.getvalue()).decode()}

def check(c,run,dev,checkpoint):
    env=make_env(c)
    obs,_=env.reset(seed=c["seed"]+100000,options={"goal":0})
    runtime=Runtime(checkpoint,dev)
    expected,_=runtime.predict(obs,123)
    del runtime
    import gc,torch
    gc.collect()
    if dev.type=="mps":
        torch.mps.empty_cache()
    elif dev.type=="cuda":
        torch.cuda.empty_cache()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port=sock.getsockname()[1]
    url=f"http://127.0.0.1:{port}"
    env_vars=os.environ.copy()
    env_vars["PYTHONPATH"]=str(ROOT)
    logpath=run/"reports"/"server.log"
    with logpath.open("w",encoding="utf-8") as log:
        child=subprocess.Popen([sys.executable,"-m","smol_workflow.p10_deployment._impl.p10_server","--checkpoint",str(checkpoint),
                                "--device",str(dev),"--port",str(port)],cwd=ROOT,env=env_vars,
                               stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if child.poll() is not None:
                    raise RuntimeError(f"HTTP 服务退出，查看 {logpath}")
                try:
                    health=request(url+"/health")
                    break
                except (urllib.error.URLError,ConnectionError):
                    time.sleep(.5)
            else:
                raise TimeoutError("HTTP 模型启动超时")
            actual=np.asarray(request(url+"/predict",payload(obs,123))["actions"],dtype=np.float32)
            if not np.allclose(expected,actual,atol=1e-5,rtol=1e-5):
                raise AssertionError("HTTP 与本地输出不一致")
            invalid=payload(obs,123)
            invalid["state"]=[float("nan"),0]
            try:
                request(url+"/predict",invalid)
                raise AssertionError("非法 state 未被拒绝")
            except urllib.error.HTTPError as exc:
                if exc.code!=400:
                    raise
            results=[]
            for goal in (0,1):
                obs,_=env.reset(seed=c["seed"]+100000,options={"goal":goal})
                frames=[obs["image"]]
                while True:
                    response=request(url+"/predict",payload(obs,100000+goal*1000+env.steps))
                    action=np.array(response["actions"][0],dtype=np.float32)
                    obs,_,done,timeout,info=env.step(action)
                    frames.append(obs["image"])
                    if done or timeout:
                        break
                results.append({"goal":goal,**info})
                save_gif(run/"reports"/f"http_goal_{goal}.gif",frames,c["fps"])
                print(f"HTTP goal={goal} {info}",flush=True)
            report={"health":health,"new_process_reloaded":True,"direct_http_equal":True,
                    "max_abs_error":float(np.abs(expected-actual).max()),"invalid_input_rejected":True,
                    "closed_loop_episodes":results,"checkpoint":str(checkpoint.relative_to(run))}
            write_json(run/"reports"/"deployment.json",report)
            return report
        finally:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
