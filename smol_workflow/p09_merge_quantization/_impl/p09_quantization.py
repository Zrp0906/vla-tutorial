r"""合并 LoRA，校验固定噪声输出，再生成教学 INT8 权重存储版本。

输入：
    - 选中的 LoRA 适配器、对应基础模型、归一化统计量与固定测试场景。
    - 只量化 lm_expert 中的 Linear 权重；输入输出投影等未选中模块保留浮点。

输出：
    - models/merged/ 完整浮点推理包；models/int8/ 含 INT8 权重、逐通道 scale 和 quantization.json。
    - reports/quantization.json 保存合并/量化差异、覆盖率与文件大小；evaluations/merged/、int8/ 保存闭环结果。
    - 采用按层反量化后浮点计算，不宣称 INT8 内核加速；合并数值对比不通过则停止。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p09_merge_quantization/09_01_merge_quantize.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本模块不解析命令行；convert(c, run, dev) 合并并量化，replace_named(policy, names) 替换指定 Linear。
    - Int8Linear(layer) 将每个输出通道权重存为 INT8；scale 和 bias 保留浮点。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p09_merge_quantization/09_01_merge_quantize.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p05_model_baseline/_impl/p05_model.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
from __future__ import annotations
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from smol_workflow.p00_pipeline._impl.p00_common import read_json, write_json
from smol_workflow.p05_model_baseline._impl.p05_model import package, load_package, load_base, Processor
from smol_workflow.p06_supervised_post_training._impl.p06_training import selected

class Int8Linear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.in_features,self.out_features = layer.in_features,layer.out_features
        weight = layer.weight.detach()
        scale = weight.abs().amax(dim=1,keepdim=True).clamp_min(1e-8)/127
        self.register_buffer("weight_int8",torch.round(weight/scale).clamp(-127,127).to(torch.int8))
        self.register_buffer("scale",scale.float())
        if layer.bias is None:
            self.register_buffer("bias",None)
        else:
            self.register_buffer("bias",layer.bias.detach().clone())

    @property
    def weight(self):
        # 官方注意力实现也会查询 weight.dtype。
        return self.weight_int8.to(self.scale.dtype) * self.scale

    def forward(self,x):
        return F.linear(x,self.weight.to(x.dtype),None if self.bias is None else self.bias.to(x.dtype))

def replace_named(policy, names):
    modules = dict(policy.named_modules())
    for name in names:
        layer = modules[name]
        if not isinstance(layer,nn.Linear):
            raise TypeError(f"{name} is not Linear")
        parent_name, _, child = name.rpartition(".")
        parent = policy.get_submodule(parent_name) if parent_name else policy
        setattr(parent,child,Int8Linear(layer))

def convert(c,run,dev):
    from peft import PeftModel
    source = selected(run,"lora")
    base = load_base(c,torch.device("cpu"))
    wrapped = PeftModel.from_pretrained(base,str(source),is_trainable=False)
    policy = wrapped.get_base_model().to(dev).eval()
    stats = read_json(source/"runtime.json")["stats"]
    processor = Processor(policy,stats,dev)
    from smol_workflow.p02_simulation._impl.p02_environment import make_env
    env = make_env(c)
    obs,_ = env.reset(seed=c["seed"]+100000,options={"goal":0})
    batch = processor.observation(obs)
    noise = torch.randn((1,c["chunk_size"],policy.config.max_action_dim),
                        generator=torch.Generator().manual_seed(100)).to(dev)
    with torch.inference_mode():
        before = policy.predict_action_chunk(batch,noise=noise.clone()).cpu()
    merged = wrapped.merge_and_unload(safe_merge=True)
    merged.eval()
    with torch.inference_mode():
        after = merged.predict_action_chunk(batch,noise=noise.clone()).cpu()
    diff = (before-after).abs()
    if not torch.allclose(before,after,atol=2e-4,rtol=2e-4):
        raise AssertionError(f"合并数值差异过大 {diff.max().item()}")
    package(merged,run/"models"/"merged",c,stats,"full")
    names = [n for n,m in merged.named_modules() if isinstance(m,nn.Linear) and "lm_expert." in n]
    params_quantized = sum(dict(merged.named_modules())[n].weight.numel() for n in names)
    total_params = sum(p.numel() for p in merged.parameters())
    replace_named(merged,names)
    with torch.inference_mode():
        quantized = merged.predict_action_chunk(batch,noise=noise.clone()).cpu()
    quant_diff = (after-quantized).abs()
    package(merged,run/"models"/"int8",c,stats,"int8")
    qinfo = {"layers":names,"quantized_weight_elements":params_quantized,"parameter_elements":total_params,
             "coverage":params_quantized/total_params,"storage":"int8 per output channel + float scale",
             "compute":"per-layer dequantize then floating F.linear; no int8 kernel",
             "merged_max_abs_error":diff.max().item(),"merged_mean_abs_error":diff.mean().item(),
             "quantized_max_abs_error":quant_diff.max().item(),"quantized_mean_abs_error":quant_diff.mean().item()}
    write_json(run/"models"/"int8"/"quantization.json",qinfo)
    qinfo["merged_weight_bytes"] = (run/"models"/"merged"/"model.safetensors").stat().st_size
    qinfo["int8_weight_bytes"] = (run/"models"/"int8"/"model.safetensors").stat().st_size
    write_json(run/"reports"/"quantization.json",qinfo)
    return qinfo
