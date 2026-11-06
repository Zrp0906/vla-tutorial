"""
功能：
    导出数据集，并用训练轨迹计算状态和动作统计量。

输入：
    <run>/data/raw/manifest.json
    <run>/data/raw/episode_XXX/trajectory.json
    <run>/data/raw/episode_XXX/帧号.png
    <run>/data/stats.json（仅 Windows 构造训练窗口时读取）

输出：
    <run>/data/stats.json（process 根据训练轨迹写出）
    <run>/data/lerobot/ 下的数据集文件
    <run>/reports/dataset.json

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    p00_cli.py
    p06_training.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from smol_workflow.p00_pipeline._impl.p00_common import read_json, write_json
from smol_workflow.p03_data_collection._impl.p03_data import episode_records

def process(c, run):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    manifest = read_json(run / "data" / "raw" / "manifest.json")
    state, action = [], []
    for e in manifest["episodes"]:
        if e["split"] == "train":
            _, rows = episode_records(run, e)
            state.extend(r["state"] for r in rows)
            action.extend(r["action"] for r in rows)
    stats = {}
    for name, values in [("state", state), ("action", action)]:
        a = np.asarray(values, dtype=np.float32)
        stats[name] = {"mean":a.mean(0).tolist(), "std":np.maximum(a.std(0), 1e-4).tolist()}
    write_json(run / "data" / "stats.json", stats)
    path = run / "data" / "lerobot"
    if path.exists():
        raise FileExistsError(path)
    features = {
        "observation.state":{"dtype":"float32", "shape":(2,), "names":["pusher_x","pusher_y"]},
        "action":{"dtype":"float32", "shape":(2,), "names":["delta_x","delta_y"]},
        "observation.images.camera1":{"dtype":"image", "shape":(256,256,3), "names":["height","width","channels"]},
    }
    ds = LeRobotDataset.create(repo_id="local/smol_push", root=path, fps=c["fps"],
                               features=features, robot_type="teaching_2d_pusher", use_videos=False)
    for e in manifest["episodes"]:
        directory, rows = episode_records(run, e)
        for t, row in enumerate(rows):
            ds.add_frame({"observation.state":np.array(row["state"], dtype=np.float32),
                          "action":np.array(row["action"], dtype=np.float32),
                          "observation.images.camera1":np.array(Image.open(directory/f"{t:04d}.png").convert("RGB")),
                          "task":e["task"]})
        ds.save_episode()
    ds.finalize()
    check = LeRobotDataset(repo_id="local/smol_push", root=path, download_videos=False)
    assert len(check) == sum(e["frames"] for e in manifest["episodes"])
    sample = check[0]
    result = {"format":"LeRobotDataset", "version":check.meta.info["codebase_version"],
              "frames":len(check), "episodes":check.meta.total_episodes,
              "sample_shapes":{k:list(v.shape) for k,v in sample.items() if isinstance(v,torch.Tensor)},
              "stats_source":"train episodes only"}
    write_json(run / "reports" / "dataset.json", result)
    return result

class Windows(Dataset):
    def __init__(self, c, run, split):
        self.c, self.run = c, Path(run)
        manifest = read_json(self.run / "data" / "raw" / "manifest.json")
        self.stats = read_json(self.run / "data" / "stats.json")
        self.samples = []
        for e in manifest["episodes"]:
            if e["split"] != split:
                continue
            directory, rows = episode_records(run, e)
            for t in range(0,len(rows),c["window_stride"]):
                self.samples.append((directory,rows,t,e["task"]))
        if not self.samples:
            raise ValueError(f"No samples for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        directory, rows, t, task = self.samples[i]
        image = np.array(Image.open(directory/f"{t:04d}.png").convert("RGB"))
        valid = min(self.c["chunk_size"], len(rows)-t)
        actions = np.zeros((self.c["chunk_size"],2), dtype=np.float32)
        actions[:valid] = np.array([r["action"] for r in rows[t:t+valid]], dtype=np.float32)
        pad = np.arange(self.c["chunk_size"]) >= valid
        state = (np.array(rows[t]["state"],dtype=np.float32)-self.stats["state"]["mean"])/self.stats["state"]["std"]
        actions = (actions-np.array(self.stats["action"]["mean"],dtype=np.float32))/np.array(self.stats["action"]["std"],dtype=np.float32)
        actions[pad] = 0
        return {"image":torch.from_numpy(image).permute(2,0,1).float()/255,
                "state":torch.tensor(state,dtype=torch.float32), "actions":torch.tensor(actions,dtype=torch.float32),
                "is_pad":torch.from_numpy(pad), "task":task}
