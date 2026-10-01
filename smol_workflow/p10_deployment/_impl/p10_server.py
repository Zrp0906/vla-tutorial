r"""在本机启动无动作队列的串行 HTTP SmolVLA 推理服务。

输入：
    - --checkpoint 指定真实推理包；仍需要项目中的基础模型/处理器资产与统计量。
    - /predict 请求包含 task、有限 state[2]、256×256 PNG 的 image_png_base64 和可选 noise_seed。

输出：
    - 仅绑定 127.0.0.1，提供 /health 与 /predict；后者返回 actions[T,2]、model 与 seconds。
    - 终端 READY 与访问日志；本模块不保存回放，部署检查才写 server.log、JSON 和 GIF。
    - 本服务持续运行直到停止，无跨请求动作队列；正常参数 --help 不加载权重也不绑定端口。

独立运行：
    - 可在真实推理包准备好后，以 -m 方式独立启动；也由 10_01 部署检查在子进程中自动调用。
    - 仅执行 10_01 检查时无需手动先启动服务器；下方例子要求 models/merged 已实际生成。

运行参数：
    - `--checkpoint`（必填；真实推理包目录，示例使用 /Users/zrp/Desktop/lora/smol/outputs/standard_run/models/merged）。
    - `--device`（默认 auto；支持 auto/cpu/cuda/mps）。
    - `--port`（整数，默认 8765）；`--help`（显示参数并退出）。

用法：
    ```bash
    cd /Users/zrp/Desktop/lora/smol
    conda run -n embodiedai python -m smol_workflow.p10_deployment._impl.p10_server --checkpoint /Users/zrp/Desktop/lora/smol/outputs/standard_run/models/merged --device auto --port 8765
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py`（模块名称引用/动态调用）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/_impl/p10_deployment.py`（模块名称引用/动态调用）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
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
