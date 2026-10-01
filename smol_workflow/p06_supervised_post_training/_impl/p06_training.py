r"""复用同一两轮训练循环进行部分参数监督训练与独立 LoRA 实验。

输入：
    - train(c, run, dev, mode)，mode 为 partial 或 lora；原始训练/验证轨迹、统计量和真实基础权重。
    - 训练 batch、学习率、LoRA 超参数从配置读取；drop_last=False 保留最后不足一批数据。

输出：
    - models/<mode>/ 的 progress.json、epoch_1/、epoch_2/、selected.json 和最终 report.json。
    - 推理包与 training_state.pt 保存优化器、调度器、CPU/打乱 RNG 状态；selected(run, mode) 返回最佳验证 checkpoint。
    - 哈希核验冻结与更新；保存训练状态但当前 SmolVLA 入口未提供 --resume，不能宣称已经支持命令行恢复。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/06_01_train_partial.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本模块没有独立命令行；06_01 调用 partial，07_01 调用 lora。
    - 两个入口均使用 --run-dir、--config、--device；epochs 必须为配置约定的 2。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/06_01_train_partial.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p09_merge_quantization/_impl/p09_quantization.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
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
