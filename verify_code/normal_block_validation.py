"""Validate the production normal-wick-block path on generated expressions.

Fixture generation is a repository adapter in this script. Once its normal
Python output has been written, the block planner receives the three final
tables plus the automatically constructed compact pairing family. The planner
certifies that family against the tables before using it for global layout
search.
The numerical target retains every open Dirac component; no ppn projection,
upper-component truncation, or substitution of a distillation VVV is used.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import opt_einsum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.block_reduction import (  # noqa: E402
    TensorSpec, TensorSymmetry, block_reduce, evaluate_expression,
    evaluate_plan, load_symbolic_output,
)
from kernels.block_reduction.canonical import clear_cache  # noqa: E402
from kernels.block_reduction.output_adapter import pairing_family_from_operators  # noqa: E402


def generate_normal_fixture(example: str, path: Path):
    """Use the real normal construction/export API outside the block core."""
    from kernels.wick_reducer import expression_output, operator_model

    operators = tuple(replace(operator, mode="normal")
                      for operator in operator_model.operator_library()[example])
    started = perf_counter()
    exported = expression_output.write_output(
        expression_output.split_operators(operators), path,
        name=f"normal_{example}", mode="independent",
    )
    return {
        "operator_mode": "normal", "raw_wick_count": exported["wick_count"],
        "independent_count": exported["independent_count"],
        "exported_view_count": exported["reduced_term_count"],
        "generation_seconds": perf_counter() - started,
    }


def _epsilon_color():
    epsilon = np.zeros((3, 3, 3))
    for p in itertools.permutations(range(3)):
        sign = (-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3))
        epsilon[p] = sign
    return epsilon


def tensor_environment(source: str):
    """Declare actual normal arrays and exact symmetries from the adapter.

    TensorSpec is explicit: this never treats a VVV/VDV operand as a normal
    epsilon/delta.  Matrix transpose symmetries are verified on the actual
    registered Dirac arrays before being declared.
    """
    from kernels import correlator_wick_normalizer as core
    from kernels.wick_reducer.expression_output import _gamma_symbol

    arrays = {_gamma_symbol(name): np.asarray(matrix, dtype=np.complex128)
              for name, matrix in core._euclidean_gamma_matrices().items()}
    arrays["epsilon_color"] = _epsilon_color()
    arrays["delta_color"] = np.eye(3)
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "contract":
            if any(not isinstance(operand, ast.Name) for operand in node.args[1:]):
                raise ValueError("normal fixture requires named tensor operands")
            names.update(operand.id for operand in node.args[1:])
    specs = {}
    for name in sorted(names):
        if name.startswith("peram_"):
            if "_x_x" not in name:
                raise ValueError("a normal fixture must expose explicit unsmeared peram endpoints")
            specs[name] = TensorSpec(
                name, (3, 4, 3, 4), ("color", "spin", "color", "spin"),
                kind="peram", peram_ports=((0, 1), (2, 3)),
            )
        elif name == "epsilon_color":
            specs[name] = TensorSpec(
                name, (3, 3, 3), ("color",) * 3,
                symmetries=(TensorSymmetry((1, 0, 2), -1),
                            TensorSymmetry((1, 2, 0), 1)),
            )
        elif name == "delta_color":
            specs[name] = TensorSpec(
                name, (3, 3), ("color",) * 2,
                symmetries=(TensorSymmetry((1, 0), 1),),
            )
        elif name in arrays:
            array = arrays[name]
            symmetries = ()
            if np.array_equal(array.T, -array):
                symmetries = (TensorSymmetry((1, 0), -1),)
            elif np.array_equal(array.T, array):
                symmetries = (TensorSymmetry((1, 0), 1),)
            specs[name] = TensorSpec(name, (4, 4), ("spin", "spin"), symmetries=symmetries)
        else:
            raise ValueError(f"operand {name!r} is not an actual normal-mode tensor")
    for name, spec in specs.items():
        if name in arrays:
            for symmetry in spec.symmetries:
                if not np.array_equal(arrays[name].transpose(symmetry.permutation),
                                      symmetry.sign * arrays[name]):
                    raise AssertionError(f"declared tensor symmetry is false for {name}")
    return specs, {name: arrays[name] for name in names if name in arrays}


def numerical_arrays(specs, constant_arrays, seed):
    rng = np.random.default_rng(seed)
    arrays = dict(constant_arrays)
    for name, spec in specs.items():
        if spec.kind == "peram":
            arrays[name] = (rng.standard_normal(spec.shape)
                            + 1j * rng.standard_normal(spec.shape)) / np.sqrt(2)
    return arrays


def _residual(expected, actual):
    if expected.shape != actual.shape:
        raise AssertionError(f"output shape changed from {expected.shape} to {actual.shape}")
    difference = expected - actual
    absolute = float(np.max(np.abs(difference))) if difference.size else 0.0
    scale = max(float(np.max(np.abs(expected))), float(np.max(np.abs(actual))), 1e-30)
    norm_scale = max(float(np.linalg.norm(expected)), float(np.linalg.norm(actual)), 1e-30)
    return {"maximum_absolute": absolute, "maximum_relative": absolute / scale,
            "norm_relative": float(np.linalg.norm(difference)) / norm_scale}


def _counted_backend():
    calls = []

    def contract(*args, **kwargs):
        calls.append(1)
        kwargs.setdefault("optimize", "greedy")
        return opt_einsum.contract(*args, **kwargs)

    return contract, calls


def audit(example="ppn", *, directory: Path, seeds=(20261013, 20261014, 20261015),
          search_seconds=10.0, search_budget=40, max_bridge_steps=0):
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / f"normal_{example}_independent.py"
    generated = generate_normal_fixture(example, source_path)
    source = source_path.read_text(encoding="utf-8")
    specs, constants = tensor_environment(source)
    parsed_at = perf_counter()
    loaded = load_symbolic_output(source, tensor_specs=specs)
    from kernels.wick_reducer.operator_model import operator_library
    operators = tuple(replace(operator, mode="normal")
                      for operator in operator_library()[example])
    family = pairing_family_from_operators(
        operators, tensor_specs=loaded.tensor_specs, compact=True,
    )
    parse_seconds = perf_counter() - parsed_at
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    nonidentity = sum(view.permutation != tuple(range(len(view.permutation)))
                      for view in loaded.basis.values())
    noninvolutive = sum(tuple(p[p[i]] for i in range(len(p))) != tuple(range(len(p)))
                       for p in (view.permutation for view in loaded.basis.values()))
    clear_cache()
    planned_at = perf_counter()
    plan = block_reduce(
        loaded.basis_contract, loaded.basis, loaded.coefficients,
        tensor_specs=loaded.tensor_specs, max_w_types=1,
        search_seconds=search_seconds, search_budget=search_budget,
        max_bridge_steps=max_bridge_steps,
        pairing_family=family,
    )
    planning_seconds = perf_counter() - planned_at
    samples = []
    for sample_index, seed in enumerate(seeds):
        arrays = numerical_arrays(specs, constants, seed)
        # Execute only source generated above by the repository's own exporter.
        # The symbolic reader and planner never execute arbitrary source files.
        source_result = None
        source_seconds = None
        if sample_index == 0:
            namespace = dict(arrays)
            source_at = perf_counter()
            exec(compile(source, str(source_path), "exec"), namespace)
            source_result = namespace[f"{loaded.prefix}_result"]
            source_seconds = perf_counter() - source_at
        reference_backend, reference_calls = _counted_backend()
        reference_at = perf_counter()
        expected = evaluate_expression(loaded.basis_contract, loaded.basis,
                                       loaded.coefficients, arrays, contract=reference_backend)
        reference_seconds = perf_counter() - reference_at
        planned_backend, planned_calls = _counted_backend()
        execution_at = perf_counter()
        actual = evaluate_plan(plan, arrays, contract=planned_backend)
        execution_seconds = perf_counter() - execution_at
        residual = _residual(expected, actual)
        parsed_residual = _residual(source_result, expected) if source_result is not None else None
        if residual["maximum_relative"] > 1e-10 or (parsed_residual and parsed_residual["maximum_relative"] > 1e-10):
            raise AssertionError(f"full-open tensor mismatch at seed {seed}: {residual}, {parsed_residual}")
        samples.append({
            "seed": seed, "output_shape": list(expected.shape),
            "output_norm": float(np.linalg.norm(expected)),
            "source_execution_seconds": source_seconds,
            "source_vs_parsed": parsed_residual,
            "reference_calls": len(reference_calls), "planned_calls": len(planned_calls),
            "reference_seconds": reference_seconds, "planned_seconds": execution_seconds,
            "residual": residual,
        })
    isolated = subprocess.run(
        [sys.executable, "-B", "-c", "import sys; from kernels import block_reduction; assert not any(n == 'kernels.wick_reducer' or n.startswith('kernels.wick_reducer.') or n == 'kernels.correlator_wick_normalizer' for n in sys.modules)"],
        cwd=PROJECT_ROOT, text=True, capture_output=True,
    )
    if isolated.returncode:
        raise AssertionError(f"production block import loaded Wick dependencies: {isolated.stderr}")
    return {
        "example": example, "target": "complete open-spin tensor", "source_path": str(source_path),
        "raw_pairing_count": len(family.terms),
        "source_sha256": source_digest, "generator": generated,
        "basis_contract_count": len(loaded.basis_contract), "basis_view_count": len(loaded.basis),
        "coefficient_count": len(loaded.coefficients),
        "coefficient_histogram": dict(sorted(Counter(map(str, loaded.coefficients)).items())),
        "nonidentity_views": nonidentity, "noninvolutive_views": noninvolutive,
        "parse_seconds": parse_seconds, "planning_seconds": planning_seconds,
        "search_parameters": {"search_seconds": search_seconds, "search_budget": search_budget,
                              "max_w_types": 1, "max_bridge_steps": max_bridge_steps},
        "plan_stats": dict(plan.stats), "rewrites": len(plan.rewrites),
        "w_shapes": [list(definition.shape) for definition in plan.w_definitions],
        "w_elements": sum(int(np.prod(definition.shape)) for definition in plan.w_definitions),
        "samples": samples, "production_import_verified": True,
        "specs": {name: asdict(spec) for name, spec in specs.items()},
    }


def render_report(result):
    stats = result["plan_stats"]
    sample_rows = []
    for sample in result["samples"]:
        residual = sample["residual"]
        sample_rows.append(
            f"| {sample['seed']} | {sample['reference_calls']} | {sample['planned_calls']} | "
            f"{sample['reference_seconds']:.3f} | {sample['planned_seconds']:.3f} | "
            f"{residual['maximum_relative']:.3e} | {residual['norm_relative']:.3e} |"
        )
    return "\n".join([
        "# normal-wick-block：真实 normal 最终表达式验证", "",
        f"本报告由 `verify_code/normal_block_validation.py` 生成，算例为 `{result['example']}`。"
        "通过原有算符构造/输出接口实际生成 normal 模式表达式，再由生产 block 入口"
        "自动构造紧凑 raw pairing family，并由证书核对其与三表目标完全相等。", "",
        "## 输入和目标", "",
        f"- 实际生成的 normal 输入：`{result['source_path']}`。",
        f"- 源文件 SHA-256：`{result['source_sha256']}`。",
        f"- 原导出器报告：{result['generator']['raw_wick_count']} raw Wick，"
        f"{result['basis_contract_count']} 独立 `basis_contract`。",
        f"- 自动配对族：{result['raw_pairing_count']} 项；它只用于全局布局搜索，"
        "不会替换三表目标。",
        f"- 完整视图及系数：{result['basis_view_count']} / {result['coefficient_count']}。"
        f"其中 {result['nonidentity_views']} 个视图非恒等转置，{result['noninvolutive_views']} 个转置非自逆。",
        f"- 整数系数的精确计数：`{result['coefficient_histogram']}`。符号规划不对系数舍入或使用数值容差。",
        f"- 数值目标保留所有四个 Dirac 分量。输出是完整 `{tuple(result['samples'][0]['output_shape'])}` 张量，"
        "没有固定自旋投影或 upper2 截断。",
        "- normal 使用真实三色 Levi-Civita epsilon，传播子轴顺序为 "
        "`(V_sink,D_sink,V_source,D_source)=(3,4,3,4)`。声明的 Gamma/epsilon 对称性先在实际数组上精确检查。",
        "- 隔离进程确认 `from kernels import block_reduction` 不加载 `wick_reducer` 或 `correlator_wick_normalizer`。", "",
        "## 当前计划的实测结果", "",
        f"- 初始不同计算网络数：{stats['initial_network_count']}。",
        f"- 返回不同计算网络数：{stats['network_count']}；W 类型数：{stats['w_type_count']}；"
        f"局部 rewrite 数：{result['rewrites']}。",
        f"- 归约后的展开证书：`symbolic_verified={stats['symbolic_verified']}`。",
        f"- 搜索参数：`{result['search_parameters']}`。",
        f"- 搜索状态数：{stats['search_states']}；`search_complete={stats['search_complete']}`；"
        f"`budget_exhausted={stats['budget_exhausted']}`；`optimality_scope={stats['optimality_scope']}`。",
        f"- normal 输出生成耗时 {result['generator']['generation_seconds']:.3f} 秒；"
        f"AST 解析 {result['parse_seconds']:.3f} 秒；计划总耗时 {result['planning_seconds']:.3f} 秒。",
        "- 搜索秒数限制不包含输入规范化与证书展开验证，因此总耗时可超过 `search_seconds`。"
        "返回计划只证明等价，不把预算内最优候选当成全局最优。", "",
        "## 完整张量数值核对", "",
        "每个 seed 独立生成随机复传播子；先按原三表计算完整目标，再执行返回计划。"
        "第一个 seed 还直接执行实际生成的 Python 源，核对 AST 读入和三表解释。"
        "下表的 planned 调用数只计完整网络收缩；W 构建由 NumPy 外积实现，未混入该计数。", "",
        "| Seed | 原三表 contract 调用 | 计划 contract 调用 | 原三表秒数 | 计划秒数 | 最大相对误差 | 范数相对误差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |", *sample_rows, "",
        "## 解释范围", "",
        "本次计数是最终 target 的不同非零完整网络数，不是最小 FLOP 或最小 wall-time 的证明。"
        "限时搜索会返回已验证的最好布局；ppn 的正式入口会自动得到 48 个网络。", "",
        "复现命令：", "", "```sh",
        f"python -B verify_code/normal_block_validation.py --example {result['example']} "
        f"--directory /private/tmp/normal_block_validation --search-seconds {result['search_parameters']['search_seconds']} "
        f"--search-budget {result['search_parameters']['search_budget']} "
        f"--max-bridge-steps {result['search_parameters']['max_bridge_steps']} "
        f"--seeds {' '.join(str(s['seed']) for s in result['samples'])}", "```", "",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("delta", "pn", "ppn", "pip"), default="ppn")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261013, 20261014, 20261015])
    parser.add_argument("--search-seconds", type=float, default=10.0)
    parser.add_argument("--search-budget", type=int, default=40)
    parser.add_argument("--max-bridge-steps", type=int, default=0)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    directory = args.directory or Path(tempfile.mkdtemp(prefix="normal_block_validation_"))
    result = audit(args.example, directory=directory, seeds=tuple(args.seeds),
                   search_seconds=args.search_seconds, search_budget=args.search_budget,
                   max_bridge_steps=args.max_bridge_steps)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(serialized + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(result), encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
