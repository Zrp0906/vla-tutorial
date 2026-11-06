"""
功能：
    启动本机推理服务，核对直接调用、非法输入和闭环控制。

输入：
    <run>/models/merged/config.json
    <run>/models/merged/runtime.json
    <run>/models/merged/model.safetensors
    项目 models/processor/ 下的处理器文件

输出：
    <run>/reports/server.log
    <run>/reports/deployment.json
    <run>/reports/http_goal_0.gif
    <run>/reports/http_goal_1.gif

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    p00_cli.py
"""
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
