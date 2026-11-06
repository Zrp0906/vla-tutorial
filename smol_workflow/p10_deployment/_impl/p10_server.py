"""
功能：
    在 127.0.0.1 提供 /health 和 /predict，按当前观测返回动作。

输入：
    --checkpoint 目录中的 config.json
    --checkpoint 目录中的 runtime.json
    --checkpoint 目录中的 model.safetensors 或 adapter_model.safetensors
    --checkpoint 目录中的 quantization.json（INT8 包）
    项目 models/base/ 下的权重（基础包或适配器包）
    项目 models/processor/ 下的处理器文件

输出：
    无

运行：
    可独立运行，也可由 p10_deployment.py 启动。

参数：
    --checkpoint：推理包目录，必填
    --device：auto、cpu、cuda 或 mps，默认 auto
    --port：监听端口，默认 8765

被谁依赖：
    00_01_check_structure.py
    p10_deployment.py
"""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
import numpy as np
from PIL import Image
from smol_workflow.p00_pipeline._impl.p00_common import device
from smol_workflow.p05_model_baseline._impl.p05_model import Runtime
from smol_workflow.p02_simulation._impl.p02_environment import TASKS

def validate_request(payload):
    if payload.get("task") not in TASKS:
        raise ValueError("task 必须是定义的推块指令")
    state=np.asarray(payload["state"],dtype=np.float32)
    if state.shape!=(2,) or not np.isfinite(state).all() or np.any((state<0)|(state>1)):
        raise ValueError("state 必须为 [0,1] 内的两个有限数")
    binary=base64.b64decode(payload["image_png_base64"],validate=True)
    with Image.open(io.BytesIO(binary)) as img:
        if img.size!=(256,256) or img.format!="PNG":
            raise ValueError("图像必须为 256×256 PNG")
        image=np.array(img.convert("RGB"))
    return {"state":state,"image":image,"task":payload["task"]}

def serve(folder,dev,port):
    runtime=Runtime(folder,dev)
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,data):
            body=json.dumps(data,allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if self.path=="/health":
                self.reply(200,{"ready":True,"model":Path(folder).name,"device":str(dev)})
            else:
                self.reply(404,{"error":"not found"})
        def do_POST(self):
            if self.path!="/predict":
                self.reply(404,{"error":"not found"})
                return
            try:
                size=int(self.headers.get("Content-Length","0"))
                if not 0<size<=2*1024*1024:
                    raise ValueError("请求体为空或超过 2 MiB")
                payload=json.loads(self.rfile.read(size))
                obs=validate_request(payload)
                chunk,seconds=runtime.predict(obs,int(payload.get("noise_seed",0)))
                self.reply(200,{"actions":chunk.tolist(),"model":Path(folder).name,"seconds":seconds})
            except (ValueError,KeyError,TypeError) as exc:
                self.reply(400,{"error":str(exc)})
            except Exception as exc:
                self.reply(500,{"error":f"{type(exc).__name__}: {exc}"})
    print(f"READY http://127.0.0.1:{port}",flush=True)
    HTTPServer(("127.0.0.1",port),Handler).serve_forever()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--device",default="auto")
    p.add_argument("--port",type=int,default=8765)
    a=p.parse_args()
    serve(a.checkpoint,device(a.device),a.port)
