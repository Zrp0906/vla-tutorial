r"""定义二维推块的 Gymnasium 环境、RGB 渲染和脚本专家。

输入：
    - PushEnv(max_steps=200, image_size=256, step_scale=0.04) 或 make_env(c)。
    - reset(seed, options={'goal':0或1})；step(action) 接收有限二维连续动作，裁剪到 [-1,1] 并按 step_scale 缩放。

输出：
    - 观测 state[2]、image[H,W,3] uint8 和 task 字符串；step 返回观测、奖励、终止/超时标记与 info。
    - 方块完整进入指定区域且保持 5 步为成功；Expert.action() 返回专家二维动作，不在此模块直接保存文件。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p02_simulation/02_01_check_simulation.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本文件无命令行参数；构造与 reset/step 参数见输入说明。
    - 在 02_01 入口进行两条指令专家检查，--run-dir 必填。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p02_simulation/02_01_check_simulation.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p02_simulation/_impl/p02_checks.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p03_data_collection/_impl/p03_data.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p08_evaluation/_impl/p08_evaluation.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p09_merge_quantization/_impl/p09_quantization.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/_impl/p10_deployment.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/_impl/p10_server.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
# 内部实现：由同阶段入口调用，不是独立执行步骤。
# 按工作流阶段组织，入口与实现同目录。
from __future__ import annotations
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

TASKS = [
    "Push the red block into the blue target zone.",
    "Push the red block into the green target zone.",
]

class PushEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(self, max_steps=200, image_size=256, step_scale=0.04):
        self.max_steps, self.image_size, self.step_scale = max_steps, image_size, step_scale
        self.action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(0, 1, (2,), dtype=np.float32),
            "image": gym.spaces.Box(0, 255, (image_size, image_size, 3), dtype=np.uint8),
            "task": gym.spaces.Text(max_length=100, charset=set("".join(TASKS))),
        })
        self.render_mode = "rgb_array"
        self.block_half = 0.035
        self.pusher_half = 0.025
        self.targets = np.array([[0.18, 0.5], [0.82, 0.5]], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.goal = int((options or {}).get("goal", 0))
        if self.goal not in (0, 1):
            raise ValueError("goal must be 0 or 1")
        self.block = np.array([self.np_random.uniform(0.43, 0.57), 0.5], dtype=np.float32)
        self.pusher = np.array([self.np_random.uniform(0.32, 0.68),
                                self.np_random.uniform(0.23, 0.33)], dtype=np.float32)
        self.steps, self.hold = 0, 0
        return self.observe(), self.info()

    def observe(self):
        return {"state": self.pusher.copy(), "image": self.render(), "task": TASKS[self.goal]}

    def info(self):
        return {"success": self.hold >= 5, "distance": float(np.linalg.norm(self.block-self.targets[self.goal])),
                "steps": self.steps}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("动作必须为有限的二维向量")
        delta = np.clip(a, -1, 1) * self.step_scale / 8
        for _ in range(8):
            candidate = np.clip(self.pusher + delta, self.pusher_half, 1-self.pusher_half)
            if np.all(np.abs(candidate - self.block) < self.pusher_half + self.block_half):
                moved = np.clip(self.block + candidate-self.pusher, self.block_half, 1-self.block_half)
                # 边界处不能穿透方块。
                if not np.allclose(moved-self.block, candidate-self.pusher, atol=1e-6):
                    candidate = self.pusher
                else:
                    self.block = moved
            self.pusher = candidate
        self.steps += 1
        inside = np.all(np.abs(self.block-self.targets[self.goal]) + self.block_half <= 0.1)
        self.hold = self.hold + 1 if inside else 0
        info = self.info()
        reward = float(info["success"]) - info["distance"]
        return self.observe(), reward, info["success"], self.steps >= self.max_steps, info

    def render(self):
        n = self.image_size
        image = Image.new("RGB", (n, n), (238, 238, 232))
        draw = ImageDraw.Draw(image)
        for center, color in zip(self.targets, [(80, 125, 245), (65, 185, 95)]):
            x, y = center*n
            draw.rectangle((x-.1*n, y-.1*n, x+.1*n, y+.1*n), fill=color)
        x, y = self.block*n
        h = self.block_half*n
        draw.rectangle((x-h, y-h, x+h, y+h), fill=(225, 55, 50), outline=(120, 30, 25))
        x, y = self.pusher*n
        h = self.pusher_half*n
        draw.ellipse((x-h, y-h, x+h, y+h), fill=(45, 45, 50))
        return np.array(image, dtype=np.uint8)

class Expert:
    def __init__(self, env):
        self.env = env
        direction = -1 if env.goal == 0 else 1
        behind = env.block + np.array([-direction*.09, 0], dtype=np.float32)
        self.waypoints = [
            np.array([behind[0], env.block[1]-.14], dtype=np.float32),
            behind,
        ]
        self.index = 0

    def action(self):
        env = self.env
        if self.index < len(self.waypoints):
            target = self.waypoints[self.index]
            if np.linalg.norm(target-env.pusher) < .006:
                self.index += 1
                return self.action()
            return np.clip((target-env.pusher)/env.step_scale, -1, 1).astype(np.float32)
        if np.all(np.abs(env.block-env.targets[env.goal]) + env.block_half <= .1):
            return np.zeros(2, dtype=np.float32)
        direction = -1 if env.goal == 0 else 1
        return np.array([direction, 0], dtype=np.float32)

def make_env(c):
    return PushEnv(c["max_steps"], c["image_size"], c["step_scale"])
