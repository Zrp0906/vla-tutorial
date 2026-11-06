# 10｜HTTP 服务与闭环控制

[全部教程](/Users/zrp/Desktop/lora/smol/README.md) · [上一篇：09｜权重合并与 INT8 量化](/Users/zrp/Desktop/lora/smol/smol_workflow/p09_merge_quantization/09_权重合并与INT8量化.md) · [下一篇：11｜实验总结与下一步改进](/Users/zrp/Desktop/lora/smol/smol_workflow/p11_feedback_summary/11_实验总结与下一步改进.md)

## 这一部分介绍什么

“部署”是让另一个程序能够使用模型，而不只是训练脚本里面自己调用。这里把模型放进一个本机 HTTP 服务：客户端递交画面和指令，服务返回动作。

可以想成小窗口：递作业进去，答案从窗口出来。HTTP 是双方递交信息的规则；本阶段只验证本机服务与仿真，不是给真实机械臂上线。

## 使用说明与前提

先完成阶段 09，确保 `models/merged/` 能加载。

```bash
cd /Users/zrp/Desktop/lora
conda run -n embodiedai python smol/smol_workflow/p10_deployment/10_01_check_deployment.py --config smol/configs/standard.json --run-dir smol/outputs/lesson_run --device auto
```

脚本自动找本机端口、启动服务子进程、执行检查，最后关闭它。你不需要先手动启动服务器。它不是长期后台服务安装命令。

## 输入和输出

输入：合并后的浮点模型包、环境观察与任务指令。

输出：`reports/deployment.json`、`server.log`、`http_goal_0.gif`、`http_goal_1.gif`。

检查包括：新进程重载、健康接口、本地与 HTTP 输出一致、非法输入拒绝、两种目标的 HTTP 闭环。两场闭环结束，不一定代表两场都成功；具体看报告中的 `closed_loop_episodes`。

## 原理：模型服务收发什么

| 项目 | 含义 | 当前要求 |
| --- | --- | --- |
| `GET /health` | 问服务是否准备好 | 返回 ready、模型名、设备 |
| `POST /predict` | 请求预测动作 | 提交下面四个字段 |
| `state` | 推动器 x、y | 0 到 1 范围内的两个有限数字 |
| `task` | 希望推向哪个目标 | 阶段 05 中定义的两条英文指令之一 |
| `image_png_base64` | 图片装进文字袋子 | 256×256 PNG 的 Base64 编码 |
| `noise_seed` | 随机噪声编号 | 可转成整数 |
| `actions` | 返回的动作串 | 标准配置下 50×2 个数字 |

Base64 是把图片字节编码成可放进 JSON 的文字，不是另一种图片识别模型。请求体最多 2 MiB。

HTTP 200 表示请求成功；400 表示提交的数据不合要求；404 表示地址不存在；500 表示服务内部异常。不要将“HTTP 200”理解成“方块已经推到目标”。

## 公式一：真正的闭环

$$
a_k=\operatorname{first}(\pi_\theta(o_k)),\qquad
q_{k+1}=F(q_k,a_k)
$$

`k` 是游戏第几步；`o_k` 是当前观察（画面、推动器位置、任务）；`π` 是模型；`first` 是取动作串的第一项；`q_k` 是完整世界状态，包括方块与推动器；`F` 是仿真的运动规则。

先观察，再预测，再移动，再重新观察。不能把只有两个数字的推动器状态当作整个世界状态，因为它没有直接列出方块位置。

小例子：第一项动作 `[0.5,0]`，在无阻挡时向右移动 0.02；走完后方块可能发生接触移动，所以必须看新画面，而不是继续用旧画面。

## 公式二：接口输出怎样比较

$$
|a-b|\leq 10^{-5}+10^{-5}|b|
$$

`a` 是本地输出中的某个数字，`b` 是 HTTP 输出对应的数字。这是当前检查所用的绝对/相对容差：允许很小的浮点差异，而不是允许“差一点也能推块”这种任务差异。

例如 `b=0.5`，允许误差约 `0.000015`。比较时使用同一个观察和同一个噪声种子，否则随机差异可能被误认为服务错误。

## 公式三：服务速度不等于 GIF 帧率

$$
T_{\mathrm{loop}}\approx T_{\mathrm{encode}}+T_{\mathrm{request}}+
T_{\mathrm{model}}+T_{\mathrm{decode}}+T_{\mathrm{env}},\qquad
f_{\mathrm{loop}}\approx\frac{1}{T_{\mathrm{loop}}}
$$

各 `T` 分别是编码、请求传输、模型计算、解码、环境更新的时间；`f` 是每秒最多完成多少个连续控制循环的粗略估计。

若总耗时是假设的 0.5 秒，则约 2 次/秒。这是算例，不是实测。配置 `fps=10` 是仿真记录和播放刻度，本服务没有保证 10 Hz 的实时调度。

## 检查清单与安全范围

默认只绑定 `127.0.0.1`，也就是本机；当前没有用户鉴权，也不是生产级服务器。不要直接改成公网监听就给别人使用。

确认新进程能加载、输出一致、坏输入被拒绝，再看闭环任务成绩。服务成功、模型成功和真实机器人安全，是三种不同的验收。

## 代码对照（可选）

想看“规则写在哪里”，可以阅读 [p10_deployment.py](/Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/_impl/p10_deployment.py)、[p10_server.py](/Users/zrp/Desktop/lora/smol/smol_workflow/p10_deployment/_impl/p10_server.py)。这些是入口调用的内部实现，不要按文件名逐个执行；先理解教程，再看函数会更容易。

---

[全部教程](/Users/zrp/Desktop/lora/smol/README.md) · [上一篇：09｜权重合并与 INT8 量化](/Users/zrp/Desktop/lora/smol/smol_workflow/p09_merge_quantization/09_权重合并与INT8量化.md) · [下一篇：11｜实验总结与下一步改进](/Users/zrp/Desktop/lora/smol/smol_workflow/p11_feedback_summary/11_实验总结与下一步改进.md)
