"""
功能：
    加载 SmolVLA，计算 Flow Matching 损失，打包和执行推理。

输入：
    项目 models/base/config.json
    项目 models/base/model.safetensors
    项目 models/processor/ 下的处理器文件
    项目 models/assets.json
    推理包中的 runtime.json
    推理包中的 model.safetensors 或 adapter_model.safetensors
    推理包中的 quantization.json（INT8 包）

输出：
    推理包中的 config.json
    推理包中的 runtime.json
    推理包中的 provenance.json
    推理包中的 model.safetensors 或 adapter_model.safetensors

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    p00_cli.py
    p06_training.py
    p09_quantization.py
    p10_deployment.py
    p10_server.py
"""
from __future__ import annotations
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from smol_workflow.p00_pipeline._impl.p00_common import ROOT, read_json, write_json, sha256, synchronize

def model_config(c, dev, source=None):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import PolicyFeature, FeatureType
    # 导入后才会把 smolvla 注册到配置表。检查点里的 type 由基类识别，不能直接交给 SmolVLAConfig。
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    cfg = PreTrainedConfig.from_pretrained(str(source or ROOT / "models" / "base"))
    if not isinstance(cfg, SmolVLAConfig):
        raise TypeError(f"需要 SmolVLAConfig，实际得到 {type(cfg).__name__}")
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
