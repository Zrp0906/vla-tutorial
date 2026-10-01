r"""导出 LeRobotDataset 并仅用训练轨迹计算归一化统计量。

输入：
    - data/raw/manifest.json、原始 PNG、trajectory.json 和统一配置。
    - 转换全部成功轨迹，但 state/action 均值和标准差只来自 train 轨迹。

输出：
    - data/lerobot/ 的 LeRobotDataset；data/stats.json 的训练集统计量。
    - reports/dataset.json 包含帧数、轨迹数、数据格式与实际样本形状；已有导出目录拒绝覆盖。
    - Windows 返回 image[3,256,256]、state[2]、actions[T,2]、is_pad[T] 和 task；T 为 chunk_size。
    - 窗口不跨轨迹；尾部补齐、掩码和归一化由该 Dataset 处理。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p04_data_processing/04_01_process_data.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本文件无命令行；process(c, run) 转换数据，Windows(c, run, split) 构造 train 或 validation 样本。
    - window_stride、chunk_size 从配置读取；Windows 是动作窗口 Dataset 类名，不是 Windows 操作系统专用模块。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p04_data_processing/04_01_process_data.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p06_supervised_post_training/_impl/p06_training.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
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
