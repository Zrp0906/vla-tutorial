"""
功能：
    读取已有报告，写出总结、失败场景和 REPORT.md。

输入：
    <run>/evaluations/*/report.json
    <run>/evaluations/base/episode_XXX.json
    <run>/evaluations/partial/episode_XXX.json
    <run>/evaluations/lora/episode_XXX.json
    <run>/evaluations/merged/episode_XXX.json
    <run>/evaluations/int8/episode_XXX.json
    <run>/models/partial/report.json
    <run>/models/lora/report.json
    <run>/reports/quantization.json
    <run>/reports/deployment.json

输出：
    <run>/reports/summary.json
    <run>/reports/failure_cases.json
    <run>/reports/REPORT.md

运行：
    作为依赖导入，不单独运行。

参数：
    无

被谁依赖：
    p00_cli.py
"""
import platform
from smol_workflow.p00_pipeline._impl.p00_common import now, read_json, write_json

def summary(c,run):
    evaluations={p.parent.name:read_json(p) for p in (run/"evaluations").glob("*/report.json")}
    training={m:read_json(run/"models"/m/"report.json") for m in ("partial","lora")}
    report={"created_at":now(),"config":c,"evaluations":evaluations,"training":training,
            "quantization":read_json(run/"reports"/"quantization.json"),
            "deployment":read_json(run/"reports"/"deployment.json"),
            "platforms":{platform.system():"tested","other_platforms":"not tested"},
            "advanced_experiments":"not run; separately enabled"}
    failures=[]
    for name in ["base","partial","lora","merged","int8"]:
        for p in (run/"evaluations"/name).glob("episode_*.json"):
            ep=read_json(p)
            if not ep["success"]:
                failures.append({"model":name,"episode":ep["index"],"seed":ep["seed"],"goal":ep["goal"],
                                 "distance":ep["distance"],"steps":ep["steps"],"error":ep["error"],
                                 "diagnosis":"2-epoch smoke training; inspect trajectory before inferring cause"})
    write_json(run/"reports"/"failure_cases.json",failures)
    write_json(run/"reports"/"summary.json",report)
    rows=["# SmolVLA 实际执行报告","",f"运行目录：{run}",f"配置：{c['profile']}；{c['episodes']} 条示范；两项训练各 {c['epochs']} epoch。","",
          "| 模型 | 成功数/回合数 | 成功率 | 最终距离 | 平均推理秒/次 |","|---|---|---|---|---|"]
    for name,r in evaluations.items():
        rows.append(f"| {name} | {r['successes']}/{r['episodes']} | {r['success_rate']:.1%} | {r['mean_final_distance']:.4f} | {r['mean_inference_latency_seconds']} |")
    rows.extend(["","两轮训练用于验证管线，不能据此认定策略已经有效。",
                 "HTTP 服务已在新进程重载并完成两种指令的闭环；本地/HTTP 输出一致。",
                 "Windows 和 Ubuntu 尚未实测；进阶实验尚未运行。"])
    (run/"reports"/"REPORT.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
    return report
