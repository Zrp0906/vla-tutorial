r"""验证二维推块仿真中的专家控制与实际 RGB 回放。

输入：
    - 统一仿真配置与运行目录；使用固定 seed 分别检查蓝色、绿色目标。

输出：
    - reports/simulation_goal_0.gif、simulation_goal_1.gif，图像来自真实环境状态。
    - 专家结果返回给调度器；任一任务失败即抛出异常。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p02_simulation/02_01_check_simulation.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本文件无命令行；simulation(c, run) 接收仿真配置与运行目录。
    - 通过 02_01 入口调用；可选 --config 和 --device，必填 --run-dir。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p02_simulation/02_01_check_simulation.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
from smol_workflow.p02_simulation._impl.p02_environment import make_env, Expert

def simulation(c,run):
    from smol_workflow.p00_pipeline._impl.p00_common import save_gif
    results=[]
    for goal in (0,1):
        env=make_env(c)
        obs,_=env.reset(seed=c["seed"],options={"goal":goal})
        expert=Expert(env)
        frames=[obs["image"]]
        while True:
            obs,_,done,timeout,info=env.step(expert.action())
            frames.append(obs["image"])
            if done or timeout:
                break
        if not info["success"]:
            raise AssertionError("专家任务失败")
        results.append(info)
        save_gif(run/"reports"/f"simulation_goal_{goal}.gif",frames,c["fps"])
    return {"expert_checks":results}
