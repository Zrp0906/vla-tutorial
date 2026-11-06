"""
功能：
    用同一训练循环做部分参数训练或 LoRA 训练，并选出验证集检查点。

输入：
    <run>/data/raw/manifest.json
    <run>/data/raw/episode_XXX/trajectory.json
    <run>/data/raw/episode_XXX/帧号.png
    <run>/data/stats.json
    项目 models/base/config.json
    项目 models/base/model.safetensors
    项目 models/processor/ 下的处理器文件
    项目 models/assets.json

输出：
    <run>/models/partial/progress.json
    <run>/models/partial/epoch_N/config.json
    <run>/models/partial/epoch_N/runtime.json
    <run>/models/partial/epoch_N/provenance.json
    <run>/models/partial/epoch_N/model.safetensors
    <run>/models/partial/epoch_N/training_state.pt
    <run>/models/partial/selected.json
    <run>/models/partial/report.json
    <run>/models/lora/progress.json
    <run>/models/lora/epoch_N/config.json
    <run>/models/lora/epoch_N/runtime.json
    <run>/models/lora/epoch_N/provenance.json
    <run>/models/lora/epoch_N/adapter_config.json
    <run>/models/lora/epoch_N/adapter_model.safetensors
    <run>/models/lora/epoch_N/training_state.pt
    <run>/models/lora/selected.json
    <run>/models/lora/report.json

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    p00_cli.py
    p09_quantization.py
"""
from __future__ import annotations
import math
import time
import torch
from torch.utils.data import DataLoader
from smol_workflow.p00_pipeline._impl.p00_common import seed_all, read_json, write_json, now, memory_report
from smol_workflow.p04_data_processing._impl.p04_data import Windows
from smol_workflow.p05_model_baseline._impl.p05_model import load_base, Processor, flow_loss, parameter_digest, package

from smol_workflow.p07_lora_finetuning._impl.p07_lora import attach_lora

def train(c, run, dev, mode):
    seed_all(c["seed"])
    policy = load_base(c,dev)
    adapter = None
    if mode == "lora":
        adapter, policy = attach_lora(policy,c)
    params = [p for p in policy.parameters() if p.requires_grad]
    names = [n for n,p in policy.named_parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("没有可训练参数")
    out = run / "models" / mode
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    train_ds, val_ds = Windows(c,run,"train"), Windows(c,run,"validation")
    loader = DataLoader(train_ds,batch_size=c["batch_size"],shuffle=True,num_workers=0,
                        generator=torch.Generator().manual_seed(c["seed"]),drop_last=False)
    vloader = DataLoader(val_ds,batch_size=c["batch_size"],shuffle=False,num_workers=0,drop_last=False)
    processor = Processor(policy,train_ds.stats,dev)
    lr = c["lora_lr"] if mode=="lora" else c["partial_lr"]
    opt = torch.optim.AdamW(params,lr=lr,betas=(.9,.95),weight_decay=1e-10)
    total = len(loader)*c["epochs"]
    warmup = max(1,math.ceil(total*.05))
    def scale(step):
        if step < warmup:
            return (step+1)/warmup
        return .1 + .9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,scale)
    frozen_before, trained_before = parameter_digest(policy,False), parameter_digest(policy,True)
    report = {"mode":mode, "device":str(dev), "config":c, "started_at":now(),
              "train_windows":len(train_ds),"validation_windows":len(val_ds),
              "parameters":sum(p.numel() for p in policy.parameters()),
              "trainable_parameters":sum(p.numel() for p in params),"trainable_names":names,
              "epochs":[],"updates":0,"completed":False}
    print(f"{mode}: train_windows={len(train_ds)} val_windows={len(val_ds)} updates={total}",flush=True)
    step, best = 0, float("inf")
    for epoch in range(c["epochs"]):
        policy.train()
        total_loss, seen = 0.0, 0
        started = time.perf_counter()
        for sample in loader:
            opt.zero_grad(set_to_none=True)
            batch = processor.batch(sample)
            loss = flow_loss(policy,batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("非有限 loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            size = sample["image"].shape[0]
            total_loss += loss.item()*size
            seen += size
            step += 1
            if step%10==0 or step==1:
                print(f"{mode} epoch={epoch+1}/2 step={step}/{total} loss={loss.item():.5f} grad={norm.item():.3f}",flush=True)
            report["updates"] = step
            report["current_epoch"] = epoch+1
            report["last_loss"] = float(loss.item())
            write_json(out/"progress.json",report)
        policy.eval()
        val_total, val_seen = 0.0, 0
        with torch.inference_mode():
            for vi,sample in enumerate(vloader):
                loss = flow_loss(policy,processor.batch(sample),seed=c["seed"]+vi)
                size = sample["image"].shape[0]
                val_total += loss.item()*size
                val_seen += size
        item = {"epoch":epoch+1,"train_loss":total_loss/seen,"validation_loss":val_total/val_seen,
                "samples_seen":seen,"seconds":time.perf_counter()-started,**memory_report(dev)}
        report["epochs"].append(item)
        checkpoint = out/f"epoch_{epoch+1}"
        package(policy,checkpoint,c,train_ds.stats,"adapter" if adapter else "full",adapter)
        torch.save({"epoch":epoch+1,"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),
                    "torch_rng":torch.get_rng_state(),"shuffle_rng":loader.generator.get_state()},
                    checkpoint/"training_state.pt")
        if item["validation_loss"] < best:
            best = item["validation_loss"]
            write_json(out/"selected.json",{"epoch":epoch+1,"checkpoint":checkpoint.name,
                                         "validation_loss":best,"selection":"validation only"})
        print(f"{mode}: {item}",flush=True)
    report.update(completed=True,completed_at=now(),
                  frozen_unchanged=parameter_digest(policy,False)==frozen_before,
                  trained_changed=parameter_digest(policy,True)!=trained_before)
    if not report["frozen_unchanged"] or not report["trained_changed"]:
        raise AssertionError("参数更新或冻结验收未通过")
    write_json(out/"report.json",report)
    return report

def selected(run, mode):
    item = read_json(run/"models"/mode/"selected.json")
    return run/"models"/mode/item["checkpoint"]
