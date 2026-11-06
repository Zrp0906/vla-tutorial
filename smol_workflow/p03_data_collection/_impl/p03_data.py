"""
功能：
    用专家采集成功轨迹、回放和训练/验证/测试划分。

输入：
    <run>/data/raw/episode_XXX/trajectory.json

输出：
    <run>/data/raw/episode_XXX/帧号.png
    <run>/data/raw/episode_XXX/trajectory.json
    <run>/data/raw/manifest.json
    <run>/reports/expert_goal_0.gif
    <run>/reports/expert_goal_1.gif

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    p00_cli.py
    p04_data.py
"""
from PIL import Image
from smol_workflow.p00_pipeline._impl.p00_common import read_json, write_json, save_gif
from smol_workflow.p02_simulation._impl.p02_environment import Expert, make_env

def collect(c, run):
    root = run / "data" / "raw"
    if (root / "manifest.json").exists():
        raise FileExistsError("原始数据已存在；请使用新 run，避免覆盖")
    root.mkdir(parents=True)
    episodes, failures = [], []
    for goal in (0, 1):
        accepted = 0
        for attempt in range(500):
            seed = c["seed"] + goal*1000 + attempt
            env = make_env(c)
            obs, _ = env.reset(seed=seed, options={"goal": goal})
            expert = Expert(env)
            records, frames = [], []
            while True:
                a = expert.action()
                records.append({"state": obs["state"].tolist(), "action": a.tolist(),
                                "timestamp": env.steps/c["fps"]})
                frames.append(obs["image"])
                obs, _, done, timeout, info = env.step(a)
                if done or timeout:
                    break
            if not info["success"]:
                failures.append({"seed": seed, "goal": goal, **info})
                continue
            index = len(episodes)
            directory = root / f"episode_{index:03d}"
            directory.mkdir()
            for t, frame in enumerate(frames):
                Image.fromarray(frame).save(directory / f"{t:04d}.png")
            split = "train" if accepted < c["train_episodes"]//2 else "validation"
            episode = {"index": index, "seed": seed, "goal": goal, "task": obs["task"],
                       "split": split, "frames": len(records), "records": records, **info}
            write_json(directory / "trajectory.json", episode)
            episodes.append({k:v for k,v in episode.items() if k != "records"})
            if accepted == 0:
                save_gif(run / "reports" / f"expert_goal_{goal}.gif", frames+[obs["image"]], c["fps"])
            accepted += 1
            if accepted == c["episodes"]//2:
                break
        if accepted != c["episodes"]//2:
            raise RuntimeError("专家未收集到足够成功轨迹")
    tests = [{"seed": c["seed"]+100000+i, "goal": g} for i in range(c["test_episodes"]//2) for g in (0,1)]
    write_json(root / "manifest.json", {"episodes":episodes, "failures":failures, "tests":tests})
    return {"episodes":len(episodes), "frames":sum(e["frames"] for e in episodes),
            "train":c["train_episodes"], "validation":c["episodes"]-c["train_episodes"], "failures":failures}

def episode_records(run, e):
    directory = run / "data" / "raw" / f"episode_{e['index']:03d}"
    return directory, read_json(directory / "trajectory.json")["records"]
