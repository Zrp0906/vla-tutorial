"""
功能：
    运行仿真交互，用 REINFORCE 更新残差策略并评测保存后重载的版本。

输入：
    <run>/data/raw/manifest.json 中的独立测试场景
    <run>/data/stats.json
    基础模型或指定检查点，以及匹配的处理器和基座资产

输出：
    <run>/models/residual_rl/residual.safetensors
    <run>/models/residual_rl/report.json
    默认使用基础模型时，按需写入 <run>/models/base/ 下的运行信息和来源信息

运行：
    作为依赖导入，不单独运行。

参数：
    c、run、dev：实验配置、运行目录和设备
    checkpoint：可选的基础策略检查点目录
    episodes、max_steps、eval_episodes：训练回合数、每局步数上限和评测回合数

被谁依赖：
    p12_cli.py
"""
# 内部实现：由同阶段入口调用，不是独立执行步骤。
import numpy as np
import torch
from safetensors.torch import load_model, save_model
from torch import nn

from smol_workflow.p00_pipeline._impl.p00_common import memory_report, read_json, seed_all, write_json
from smol_workflow.p02_simulation._impl.p02_environment import make_env
from smol_workflow.p05_model_baseline._impl.p05_model import Runtime, parameter_digest

RESIDUAL_SCALE = 0.25
EXPLORATION_STD = 0.3
DISCOUNT = 0.99


def make_actor(dev):
    actor = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 2)).to(dev)
    nn.init.zeros_(actor[-1].weight)
    nn.init.zeros_(actor[-1].bias)
    return actor


def episode(runtime, c, case, actor=None, optimizer=None):
    env = make_env(c)
    obs, info = env.reset(seed=case["seed"], options={"goal": case["goal"]})
    runtime.reset()
    rewards, log_probs = [], []
    distance = info["distance"]
    approach = float(np.linalg.norm(env.pusher - env.block))
    noise_seed = c["eval_noise_seed"] + (case["seed"] * 2 + case["goal"]) * env.max_steps
    while True:
        chunk, _ = runtime.predict(obs, noise_seed=noise_seed + env.steps)
        action = np.clip(chunk[0], -1, 1)
        if actor is not None:
            features = np.concatenate((obs["state"], env.block, env.targets[env.goal], action))
            mean = actor(torch.as_tensor(features, dtype=torch.float32, device=runtime.dev))
            if optimizer is None:
                residual = mean
            else:
                distribution = torch.distributions.Normal(mean, EXPLORATION_STD)
                residual = distribution.sample()
                log_probs.append(distribution.log_prob(residual).sum())
            action = np.clip(action + RESIDUAL_SCALE * residual.detach().tanh().cpu().numpy(), -1, 1)

        obs, _, done, timeout, info = env.step(action)
        next_approach = float(np.linalg.norm(env.pusher - env.block))
        reward = 5 * (distance - info["distance"]) + 0.5 * (approach - next_approach) + float(done) - 0.01
        rewards.append(reward)
        distance, approach = info["distance"], next_approach
        if done or timeout:
            break

    result = {**case, **info, "reward": sum(rewards)}
    if optimizer is not None:
        returns = []
        discounted = 0.0
        for reward in reversed(rewards):
            discounted = reward + DISCOUNT * discounted
            returns.append(discounted)
        weights = torch.tensor(returns[::-1], device=runtime.dev, dtype=torch.float32)
        loss = -(torch.stack(log_probs) * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        result["loss"] = float(loss.detach())
    return result


def train_residual(c, run, dev, checkpoint, episodes, max_steps, eval_episodes):
    from smol_workflow.p00_pipeline._impl.p00_cli import base_package

    cases = read_json(run / "data" / "raw" / "manifest.json")["tests"][:eval_episodes]
    if len(cases) != eval_episodes or sum(case["goal"] == 0 for case in cases) != eval_episodes // 2:
        raise ValueError("运行目录中没有足够的成对测试场景")
    if checkpoint is None:
        checkpoint = run / "models" / "base"
        if not checkpoint.exists():
            checkpoint = base_package(c, run)
    runtime = Runtime(checkpoint, dev)
    runtime.policy.requires_grad_(False)
    runtime.policy.eval()
    seed_all(c["seed"])
    sim_config = {**c, "max_steps": max_steps or c["max_steps"]}
    actor = make_actor(dev)
    before = parameter_digest(actor, True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    out = run / "models" / "residual_rl"
    out.mkdir(parents=True, exist_ok=False)
    training = []
    for index in range(episodes):
        case = {"seed": c["seed"] + 200000 + index, "goal": index % 2}
        result = episode(runtime, sim_config, case, actor, optimizer)
        training.append(result)
        print(f"residual_rl episode={index+1}/{episodes} {result}", flush=True)
    if parameter_digest(actor, True) == before:
        raise AssertionError("残差策略参数没有更新")

    weights = out / "residual.safetensors"
    save_model(actor, str(weights))
    reloaded = make_actor(dev)
    load_model(reloaded, str(weights), device=str(dev))
    reloaded.eval()
    if parameter_digest(reloaded, True) != parameter_digest(actor, True):
        raise AssertionError("残差策略重载后的权重不一致")
    evaluation = {}
    with torch.inference_mode():
        for name, controller in (("base", None), ("residual", reloaded)):
            results = [episode(runtime, sim_config, case, controller) for case in cases]
            evaluation[name] = {"success_rate": sum(row["success"] for row in results) / len(results),
                                "episodes": results}
    write_json(out / "report.json", {
        "completed": True, "device": str(dev), "base_checkpoint": str(checkpoint.resolve()),
        "base_assets": runtime.meta["assets"], "config": sim_config, "residual_scale": RESIDUAL_SCALE,
        "training_steps": sum(row["steps"] for row in training), "training": training,
        "evaluation": evaluation, **memory_report(dev),
    })
