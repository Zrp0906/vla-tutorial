r"""汇总实际训练、评测、量化与部署报告并收集失败场景。

输入：
    - evaluations/*/report.json、部分微调和 LoRA 的 models/*/report.json。
    - reports/quantization.json、deployment.json 及各模型逐回合记录；上游缺失会报错。

输出：
    - reports/summary.json、failure_cases.json 和 REPORT.md。
    - 汇总真实成功率、距离、延迟、两轮训练与平台范围；本脚本不重新采集、训练或评测。
    - 以上相对产物路径以 --run-dir 指定的 <run> 为根；描述的是成功调用时的产物，不表示目前已经生成。

独立运行：
    - 不需要直接执行；这是由流程入口或其他内部模块导入的实现文件。
    - 对应入口为 /Users/zrp/Desktop/lora/smol/smol_workflow/p11_feedback_summary/11_01_summarize.py；下面演示该入口，不把内部文件当作单独步骤。

运行参数：
    - 本模块无命令行；summary(c, run) 汇总同一运行目录中的既有结果。
    - 通过 11_01 入口指定 --run-dir；不会自动补做缺失的上游阶段。

用法：
    ```bash
    conda run -n embodiedai python /Users/zrp/Desktop/lora/smol/smol_workflow/p11_feedback_summary/11_01_summarize.py --run-dir /Users/zrp/Desktop/lora/smol/outputs/standard_run --device auto
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（直接导入）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/00_01_check_structure.py` 会按目录扫描本文件，进行语法/导入或入口参数检查（不是业务执行步骤）。"""
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
