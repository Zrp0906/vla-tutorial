r"""采集两条语言指令的成功示范并保存轨迹与独立测试清单。

输入：
    - 统一配置、运行目录和脚本专家；默认 50 条成功示范，每条指令 25 条。
    - 按完整轨迹划分 40 条训练、10 条验证；另保存 20 个固定独立测试场景。

输出：
    - data/raw/episode_XXX/ 下的逐帧 PNG 与 trajectory.json。
    - data/raw/manifest.json 记录轨迹划分、失败尝试与测试 seed/goal；reports/expert_goal_0.gif、expert_goal_1.gif。
    - 原始数据目录或清单已存在时拒绝覆盖，使用新运行目录。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p03_data_collection/03_01_collect_data.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本文件无命令行；collect(c, run) 执行采集，episode_records(run, e) 读取某轨迹记录。
    - episodes、train_episodes、test_episodes、seed 与 fps 从 JSON 配置读取，不是独立命令行选项。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p03_data_collection/03_01_collect_data.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p04_data_processing/_impl/p04_data.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
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
