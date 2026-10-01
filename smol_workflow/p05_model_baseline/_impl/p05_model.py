r"""适配真实 SmolVLA 的二维输入输出，提供模型加载、Flow Matching 与推理包接口。

输入：
    - models/base/ 的真实 SmolVLA 配置和权重、models/processor/ 分词器以及训练集 state/action 统计量。
    - 训练 batch 的 CHW 图像、状态、动作块、padding 掩码与任务；推理观测为 RGB 图像、state[2]、task。

输出：
    - load_base/load_package 返回真实策略；Processor 将语言补换行并分词、迁移设备及反归一化动作。
    - flow_loss 返回仅有效时间步与外部二维动作的损失；Runtime.predict 返回环境单位 actions[T,2] 和耗时。
    - package 写配置、runtime.json、来源哈希及完整权重或适配器；所有产物需真正调用后才创建。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p05_model_baseline/05_02_evaluate_baseline.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本文件不解析命令行；load_base(c, dev)、Runtime(folder, dev)、predict(obs, noise_seed=None) 与 package(...) 是 Python 接口。
    - 先运行 05_01 下载，再用 05_02 基线入口检查；本次文档更新不表示模型已成功加载。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p05_model_baseline/05_02_evaluate_baseline.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/_impl/p06_training.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p09_merge_quantization/_impl/p09_quantization.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/_impl/p10_deployment.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/_impl/p10_server.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
from __future__ import annotations
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from smol_workflow.p00_pipeline._impl.p00_common import ROOT, read_json, write_json, sha256, synchronize

def model_config(c, dev, source=None):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    cfg = SmolVLAConfig.from_pretrained(str(source or ROOT / "models" / "base"))
    cfg.device = str(dev)
    # 完整 SmolVLA checkpoint 已包含视觉语言骨干，无需第二次下载骨干权重。
    cfg.load_vlm_weights = False
    cfg.vlm_model_name = str(ROOT / "models" / "processor")
    cfg.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        "observation.images.camera1": PolicyFeature(FeatureType.VISUAL, (3,256,256)),
    }
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (2,))}
    cfg.n_action_steps = 1
    cfg.chunk_size = c["chunk_size"]
    cfg.num_steps = c["inference_steps"]
    cfg.freeze_vision_encoder = True
    cfg.train_expert_only = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    return cfg

def load_base(c, dev):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    base = ROOT / "models" / "base"
    if not (base / "model.safetensors").exists():
        raise FileNotFoundError("请先运行 05_01_download_assets.py 完成真实权重下载")
    cfg = model_config(c, "cpu")
    policy = SmolVLAPolicy.from_pretrained(str(base), config=cfg, strict=True)
    return policy.to(dev).float()

def package(policy, folder, c, stats, kind, adapter=None):
    from safetensors.torch import save_model
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    policy.config.save_pretrained(folder)
    write_json(folder/"runtime.json", {"config":c, "stats":stats, "kind":kind,
                                      "assets":read_json(ROOT/"models"/"assets.json"),
                                      "preprocessing":"newline; CHW float [0,1]; train-only state/action mean_std",
                                      "image_key":"observation.images.camera1"})
    if adapter is not None:
        adapter.save_pretrained(folder)
    else:
        save_model(policy, str(folder/"model.safetensors"))
    weight = folder/("adapter_model.safetensors" if adapter is not None else "model.safetensors")
    write_json(folder/"provenance.json", {"kind":kind, "weight_sha256":sha256(weight),
                                        "weight_bytes":weight.stat().st_size,
                                        "base_revision":read_json(ROOT/"models"/"assets.json")["base"]["revision"]})

def load_package(folder, dev):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_model
    folder = Path(folder)
    meta = read_json(folder/"runtime.json")
    cfg = model_config(meta["config"], "cpu", None if meta["kind"] == "base" else folder)
    if meta["kind"] == "base":
        policy = load_base(meta["config"], torch.device("cpu"))
    elif meta["kind"] == "adapter":
        from peft import PeftModel
        base = load_base(meta["config"], torch.device("cpu"))
        wrapped = PeftModel.from_pretrained(base, str(folder), is_trainable=False)
        policy = wrapped.get_base_model()
    else:
        policy = SmolVLAPolicy(cfg)
        if meta["kind"] == "int8":
            from smol_workflow.p09_merge_quantization._impl.p09_quantization import replace_named
            replace_named(policy, read_json(folder/"quantization.json")["layers"])
        load_model(policy, str(folder/"model.safetensors"), strict=True)
    return policy.to(dev).float().eval(), meta

class Processor:
    def __init__(self, policy, stats, dev):
        self.tokenizer = policy.model.vlm_with_expert.processor.tokenizer
        self.stats, self.dev = stats, dev
        self.max_length = policy.config.tokenizer_max_length

    def batch(self, sample):
        tokens = self.tokenizer([t if t.endswith("\n") else t+"\n" for t in sample["task"]],
                                padding="max_length", max_length=self.max_length, truncation=True,
                                return_tensors="pt")
        return {"observation.images.camera1":sample["image"].to(self.dev),
                "observation.state":sample["state"].to(self.dev),
                "observation.language.tokens":tokens["input_ids"].to(self.dev),
                "observation.language.attention_mask":tokens["attention_mask"].bool().to(self.dev),
                "action":sample["actions"].to(self.dev),
                "actions_is_pad":sample["is_pad"].to(self.dev)}

    def observation(self, obs):
        state = (np.asarray(obs["state"],dtype=np.float32)-self.stats["state"]["mean"])/self.stats["state"]["std"]
        sample = {"image":torch.from_numpy(obs["image"].copy()).permute(2,0,1).float().unsqueeze(0)/255,
                  "state":torch.tensor(state,dtype=torch.float32).unsqueeze(0),
                  "actions":torch.zeros(1,1,2), "is_pad":torch.zeros(1,1,dtype=torch.bool),
                  "task":[obs["task"]]}
        return self.batch(sample)

    def unnormalize(self, action):
        mean = torch.tensor(self.stats["action"]["mean"],dtype=action.dtype,device=action.device)
        std = torch.tensor(self.stats["action"]["std"],dtype=action.dtype,device=action.device)
        return action * std + mean

def flow_loss(policy, batch, seed=None):
    """保留官方 Flow Matching 目标，只在有效时间和外部二维动作上归约。"""
    images, masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    b = actions.shape[0]
    noise = time_tensor = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(actions.shape,generator=generator).to(state.device)
        time_tensor = torch.rand(b,generator=generator).mul(.998).add(.001).to(state.device)
    losses = policy.model.forward(images,masks,batch["observation.language.tokens"],
                                  batch["observation.language.attention_mask"],state,actions,noise,time_tensor)
    valid = (~batch["actions_is_pad"]).unsqueeze(-1)
    effective = losses[:,:,:2]
    return (effective * valid).sum() / (valid.sum().clamp_min(1)*2)

def parameter_digest(policy, trainable):
    digest = hashlib.sha256()
    for name, value in policy.named_parameters():
        if value.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

class Runtime:
    def __init__(self, folder, dev):
        self.policy, self.meta = load_package(folder, dev)
        self.dev = dev
        self.processor = Processor(self.policy,self.meta["stats"],dev)
        self.calls = 0

    def reset(self):
        self.policy.reset()
        self.calls = 0

    def predict(self, obs, noise_seed=None):
        batch = self.processor.observation(obs)
        generator = torch.Generator(device="cpu").manual_seed(int(noise_seed or 0))
        noise = torch.randn((1,self.policy.config.chunk_size,self.policy.config.max_action_dim),
                            generator=generator).to(self.dev)
        synchronize(self.dev)
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.policy.predict_action_chunk(batch,noise=noise)
            actions = self.processor.unnormalize(out)[0].cpu().numpy()
        synchronize(self.dev)
        self.calls += 1
        if actions.ndim != 2 or actions.shape[1] != 2 or not np.isfinite(actions).all():
            raise ValueError("模型输出包含非法动作")
        return actions.astype(np.float32), time.perf_counter()-start
