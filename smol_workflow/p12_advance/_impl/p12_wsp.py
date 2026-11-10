"""
功能：
    用配对闭环场景诊断同义指令故障，验证语言嵌入的直接权重补丁。

输入：
    运行目录中的测试清单和统计量、浮点模型及本地处理器
    对照阶段另读诊断结果和大模型给出的 JSON 补丁

输出：
    diagnosis.json、patch_template.json、experiment.json 和诊断回放
    comparison/ 下的可重放补丁、闭环回放和对照报告

运行：
    作为依赖导入，不单独运行。

参数：
    由 p12_cli.py 传入实验目录、模型、设备和补丁。

被谁依赖：
    p12_cli.py
"""
# 内部实现：由同阶段入口调用，不是独立执行步骤。
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from smol_workflow.p00_pipeline._impl.p00_common import ROOT, device, read_json, sha256, write_json
from smol_workflow.p02_simulation._impl.p02_environment import TASKS
from smol_workflow.p05_model_baseline._impl.p05_model import Runtime
from smol_workflow.p08_evaluation._impl.p08_evaluation import evaluate

COLORS = ("blue", "green")
DEVELOPMENT = "Push the red block into the {color} area."
HELDOUT = "Please push the red block into the {color} area."
MAX_COORDINATES = 32


def paired_cases(tests, c, seeds_per_split):
    goals_by_seed = {}
    for case in tests:
        goals_by_seed.setdefault(case["seed"], set()).add(case["goal"])
    seeds = sorted(seed for seed, goals in goals_by_seed.items() if goals == {0, 1})
    if len(seeds) < 2 * seeds_per_split:
        raise ValueError("测试清单没有足够的成对蓝区和绿区种子，请减小 --seeds-per-split")

    splits = []
    for split, template in enumerate((DEVELOPMENT, HELDOUT)):
        cases = []
        start = split * seeds_per_split
        for index, seed in enumerate(seeds[start:start + seeds_per_split], start=start):
            for goal, color in enumerate(COLORS):
                noise = c["eval_noise_seed"] + (index * len(COLORS) + goal) * c["max_steps"]
                case = {"seed": seed, "goal": goal, "noise_seed": noise}
                cases.extend((
                    {**case, "wording": "original", "task": TASKS[goal]},
                    {**case, "wording": "paraphrase", "task": template.format(color=color)},
                ))
        splits.append(cases)
    return splits


def source_identity(checkpoint):
    meta = checkpoint / "runtime.json"
    kind = read_json(meta)["kind"]
    if kind not in ("base", "full"):
        raise ValueError("WSP 需要基础模型或完整浮点推理包，例如已合并的 LoRA 模型")
    weights_dir = ROOT / "models" / "base" if kind == "base" else checkpoint
    processor = ROOT / "models" / "processor"
    return {
        "runtime": sha256(meta),
        "config": sha256(weights_dir / "config.json"),
        "weights": sha256(weights_dir / "model.safetensors"),
        "processor": {file.name: sha256(file) for file in sorted(processor.iterdir()) if file.is_file()},
    }


def input_embedding(runtime):
    text_model = runtime.policy.model.vlm_with_expert.get_vlm_model().text_model
    weight = text_model.get_input_embeddings().weight
    name = next(name for name, param in runtime.policy.named_parameters() if param is weight)
    return name, weight


def diagnosis_tokens(runtime, cases, weight):
    tokenizer = runtime.processor.tokenizer
    tokens = {}
    for word in ("area", "zone", "target", "region"):
        ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
        if len(ids) != 1 or ids[0] in tokenizer.all_special_ids or not 0 <= ids[0] < len(weight):
            raise ValueError(f"无法将 {word} 定位为一行语言嵌入")
        tokens[word] = {"token_id": ids[0], "weights": weight[ids[0]].detach().cpu().tolist()}
    if len({token["token_id"] for token in tokens.values()}) != len(tokens):
        raise ValueError("候选词元对应重复的嵌入行")
    for case in cases:
        ids = tokenizer(case["task"] + "\n", truncation=True,
                        max_length=runtime.processor.max_length)["input_ids"]
        paraphrase = case["wording"] == "paraphrase"
        if (tokens["area"]["token_id"] in ids) != paraphrase:
            raise ValueError("实际推理文本中的 area 词元与预期不符")
        if not paraphrase and tokens["zone"]["token_id"] not in ids:
            raise ValueError("原指令未包含 zone 词元")
    return tokens


def rollouts(c, staged, label, runtime):
    count = evaluate(c, staged, label, runtime)["episodes"]
    folder = staged / "evaluations" / label
    return [read_json(folder / f"episode_{index:03d}.json") for index in range(count)]


def prepare(c, run, out, dev, checkpoint, seeds_per_split):
    if out.exists():
        raise FileExistsError(out)
    tests = read_json(run / "data" / "raw" / "manifest.json")["tests"]
    development, heldout = paired_cases(tests, c, seeds_per_split)
    if checkpoint is None:
        from smol_workflow.p00_pipeline._impl.p00_cli import base_package

        checkpoint = run / "models" / "base"
        if not (checkpoint / "runtime.json").is_file():
            checkpoint = base_package(c, run)
    checkpoint = checkpoint.resolve()
    source = source_identity(checkpoint)
    runtime = Runtime(checkpoint, dev)
    name, weight = input_embedding(runtime)
    tokens = diagnosis_tokens(runtime, development + heldout, weight)

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".wsp-", dir=out.parent) as temporary:
        staged = Path(temporary)
        write_json(staged / "data" / "raw" / "manifest.json", {"tests": development})
        results = rollouts(c, staged, "diagnosis", runtime)
        gaps = [sum(original["goal"] == goal and original["success"] and not paraphrase["success"]
                    for original, paraphrase in zip(results[::2], results[1::2]))
                for goal in range(len(COLORS))]
        diagnosis = {
            "parameter": name, "editable_token": "area", "tokens": tokens,
            "original_success_paraphrase_failure": dict(zip(COLORS, gaps)),
            "reproduced": max(gaps) >= 2,
            "episodes": [{key: row[key] for key in
                          ("seed", "goal", "task", "wording", "success", "steps", "distance")}
                         for row in results],
        }
        diagnosis_path = staged / "diagnosis.json"
        write_json(diagnosis_path, diagnosis)
        fingerprint = sha256(diagnosis_path)
        write_json(staged / "patch_template.json", {
            "diagnosis_sha256": fingerprint, "reference_token_id": None,
            "strength": None, "reason": "",
        })
        write_json(staged / "experiment.json", {
            "checkpoint": str(checkpoint), "source": source, "diagnosis_sha256": fingerprint,
            "device": str(dev), "config": c, "heldout_cases": heldout,
        })
        staged.rename(out)
    return out / "diagnosis.json"


def patch_row(weight, diagnosis, patch, fingerprint):
    if not isinstance(patch, dict) or patch.get("diagnosis_sha256") != fingerprint:
        raise ValueError("补丁的 diagnosis_sha256 与本次诊断不一致")
    common = {"diagnosis_sha256", "reason"}
    if not isinstance(patch.get("reason"), str) or not patch["reason"].strip():
        raise ValueError("补丁需要提供非空的 reason")
    tokens = diagnosis["tokens"]
    token_id = tokens["area"]["token_id"]
    original = weight[token_id].detach().clone()
    if set(patch) == common | {"reference_token_id", "strength"}:
        reference = patch["reference_token_id"]
        strength = patch["strength"]
        refs = {tokens[word]["token_id"] for word in ("zone", "target", "region")}
        if type(reference) is not int or reference not in refs:
            raise ValueError("reference_token_id 必须是诊断中列出的 zone、target 或 region")
        if type(strength) not in (int, float) or not 0 < strength <= 1:
            raise ValueError("strength 需要在 (0, 1] 内")
        edited = original.lerp(weight[reference].detach(), strength)
    elif set(patch) == common | {"deltas"}:
        deltas = patch["deltas"]
        if not isinstance(deltas, dict) or not 1 <= len(deltas) <= MAX_COORDINATES:
            raise ValueError(f"deltas 需要包含 1–{MAX_COORDINATES} 个坐标")
        limit = original.norm().item()
        change = torch.zeros_like(original)
        for index, value in deltas.items():
            if not index.isascii() or not index.isdecimal() or len(index) > len(str(len(change))):
                raise ValueError("deltas 包含非法坐标")
            coordinate = int(index)
            if index != str(coordinate) or coordinate >= len(change):
                raise ValueError("deltas 包含非法坐标")
            if type(value) not in (int, float) or not -limit <= value <= limit:
                raise ValueError("deltas 包含非法坐标或修改量")
            change[coordinate] = value
        if change.norm().item() > limit:
            raise ValueError("deltas 的总改变量超过原嵌入行的 L2 范数")
        edited = original + change
    else:
        raise ValueError("补丁需要提供 reference_token_id 和 strength，或者提供 deltas")
    if not torch.isfinite(edited).all() or torch.equal(edited, original):
        raise ValueError("补丁必须产生有限且实际变化的权重")
    return edited


def compare_patch(out, patch_path, requested_device):
    result = out / "comparison"
    if result.exists():
        raise FileExistsError(result)
    experiment = read_json(out / "experiment.json")
    diagnosis_path = out / "diagnosis.json"
    fingerprint = experiment["diagnosis_sha256"]
    if sha256(diagnosis_path) != fingerprint:
        raise ValueError("诊断文件已改变，请重新准备实验")
    diagnosis = read_json(diagnosis_path)
    if not diagnosis["reproduced"]:
        raise ValueError("诊断未复现同义指令故障，请选择合适的浮点策略或新的诊断场景")

    checkpoint = Path(experiment["checkpoint"])
    if source_identity(checkpoint) != experiment["source"]:
        raise ValueError("检查点或处理器已改变，请重新准备实验")
    dev = device(requested_device or experiment["device"].split(":")[0])
    runtime = Runtime(checkpoint, dev)
    name, weight = input_embedding(runtime)
    if name != diagnosis["parameter"]:
        raise ValueError("当前模型的语言嵌入参数与诊断不一致")
    tokens = diagnosis["tokens"]
    target = tokens["area"]["token_id"]
    original = weight[target].detach().clone()
    synonym = weight[tokens["zone"]["token_id"]].detach()
    patch = read_json(patch_path)
    edited = patch_row(weight, diagnosis, patch, fingerprint)
    heldout = experiment["heldout_cases"]

    with TemporaryDirectory(prefix=".comparison-", dir=out) as temporary, torch.no_grad():
        staged = Path(temporary)
        write_json(staged / "data" / "raw" / "manifest.json", {"tests": heldout})
        write_json(staged / "patch.json", patch)
        scores = {}
        try:
            for label, row in (("original", original), ("synonym", synonym), ("wsp", edited)):
                weight[target].copy_(row)
                scores[label] = rollouts(experiment["config"], staged, label, runtime)
        finally:
            weight[target].copy_(original)

        groups = []
        for goal, color in enumerate(COLORS):
            for wording in ("original", "paraphrase"):
                selected = [i for i, case in enumerate(heldout)
                            if case["goal"] == goal and case["wording"] == wording]
                base = [scores["original"][i]["success"] for i in selected]
                wsp = [scores["wsp"][i]["success"] for i in selected]
                groups.append({
                    "goal": color, "wording": wording, "episodes": len(selected),
                    "successes": {label: sum(rows[i]["success"] for i in selected)
                                  for label, rows in scores.items()},
                    "repairs": sum(not before and after for before, after in zip(base, wsp)),
                    "regressions": sum(before and not after for before, after in zip(base, wsp)),
                })
        paraphrases = [row for row in groups if row["wording"] == "paraphrase"]
        improvement = sum(row["repairs"] - row["regressions"] for row in paraphrases)
        gain_over_synonym = sum(row["successes"]["wsp"] - row["successes"]["synonym"]
                                for row in paraphrases)
        accepted = improvement > 0 and all(
            row["regressions"] == 0 if row["wording"] == "original"
            else row["repairs"] >= row["regressions"] for row in groups
        )
        write_json(staged / "report.json", {
            "accepted": accepted, "device": str(dev), "source": experiment["source"],
            "parameter": name, "token_id": target,
            "changed_coordinates": int(torch.count_nonzero(edited != original).item()),
            "diagnosis_sha256": fingerprint, "patch_sha256": sha256(staged / "patch.json"),
            "paraphrase_success_gain": improvement, "paraphrase_gain_over_synonym": gain_over_synonym,
            "groups": groups,
        })
        staged.rename(result)
    return result / "report.json"
