"""
功能：
    用固定种子检查两条指令的专家控制，并保存 GIF。

输入：
    无

输出：
    <run>/reports/simulation_goal_0.gif
    <run>/reports/simulation_goal_1.gif

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    p00_cli.py
"""
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
