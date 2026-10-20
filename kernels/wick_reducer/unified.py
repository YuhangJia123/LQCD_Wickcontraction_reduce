"""Unified input/mode/output facade for the merged Wick reducers.

The public vocabulary is deliberately kept at the top level::

    input  = ``distillation`` (also accepts historical ``distrilation``) | ``normal``
    mode   = ``wick`` | ``special``
    output = ``independent`` | ``check`` | ``block`` | ``count``

The :mod:`wick_reducer` output writers remain available.  ``special`` uses
``wick_colour_orbit``'s incidence + global-colour S3 fused kernel, while
``wick`` uses the original graph reducer and therefore preserves its Python
contract output format.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from kernels import correlator_wick_normalizer as core

from . import operator_model
from .expression_output import split_operators, write_output as write_wick_output

# Imported lazily in the functions below so the generic Wick reducer can be
# imported without loading the colour-fusion package or its exact backend.

_INPUT_ALIASES = {
    "distillation": "distillation",
    "distrilation": "distillation",  # historical spelling in notes/scripts
    "distill": "distillation",
    "normal": "normal",
}
_MODE_ALIASES = {"wick": "wick", "special": "special"}
_OUTPUT_ALIASES = {
    "independent": "independent",
    "indenpent": "independent",  # historical typo in the project notes
    "check": "check",
    "block": "block",
    "count": "count",
}


@dataclass(frozen=True)
class UnifiedConfig:
    input: str = "distillation"
    mode: str = "wick"
    output: str = "independent"
    name: str = "correlator"
    wick_scope: str = "complete"
    normal_ordered: bool = False
    isospin_degenerate_perams: bool = True
    tolerance: float = 1.0e-12
    batch_size: int = 4096

    def __post_init__(self) -> None:
        # Store canonical spellings so direct API callers get the same result
        # directory names as JSON/CLI callers.
        object.__setattr__(self, "input", canonical_input(self.input))
        object.__setattr__(self, "mode", canonical_mode(self.mode))
        object.__setattr__(self, "output", canonical_output(self.output))
        if self.wick_scope not in {"complete", "cross_time"}:
            raise ValueError("wick_scope must be 'complete' or 'cross_time'")
        if type(self.normal_ordered) is not bool:
            raise ValueError("normal_ordered must be true or false")
        if type(self.isospin_degenerate_perams) is not bool:
            raise ValueError("isospin_degenerate_perams must be true or false")
        if not math.isfinite(float(self.tolerance)) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.mode == "special" and self.input != "normal":
            raise ValueError("mode='special' is available only for normal input")
        if self.output == "count" and self.mode != "special":
            raise ValueError("output='count' is available only for special mode")
        if self.output == "block" and not (self.mode == "wick" and self.input == "normal"):
            raise ValueError("output='block' is available only for normal-wick input")


def canonical_input(value: str) -> str:
    try:
        return _INPUT_ALIASES[str(value).strip().lower()]
    except KeyError as exc:
        raise ValueError("input must be 'distillation' or 'normal'") from exc


def canonical_mode(value: str) -> str:
    try:
        return _MODE_ALIASES[str(value).strip().lower()]
    except KeyError as exc:
        raise ValueError("mode must be 'wick' or 'special'") from exc


def canonical_output(value: str) -> str:
    try:
        return _OUTPUT_ALIASES[str(value).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "output must be 'independent', 'check', 'block', or 'count'"
        ) from exc


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def result_directory(config: UnifiedConfig, root: str | Path = "result") -> Path:
    """Return and create ``result/input_mode_output``."""
    path = Path(root) / f"{config.input}_{config.mode}_{config.output}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _operator_records(document: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    raw = document.get("operators")
    if not isinstance(raw, list):
        raise ValueError("input.operators must be an array")
    if len(raw) == 2 and all(isinstance(item, list) for item in raw):
        sink, source = raw
    else:
        sink = [item for item in raw if isinstance(item, Mapping) and item.get("side") == "sink"]
        source = [item for item in raw if isinstance(item, Mapping) and item.get("side") == "source"]
    if not sink or not source:
        raise ValueError("operators must contain nonempty sink and source lists")
    if any(not isinstance(item, Mapping) for item in (*sink, *source)):
        raise ValueError("operator records must be JSON objects")
    return list(sink), list(source)


def _make_operator(record: Mapping[str, Any], side: str, mode: str, index: int):
    declared_side = record.get("side", side)
    if declared_side != side:
        raise ValueError(f"operators[{side}][{index}].side must be {side!r}")
    required = ("kind", "species", "flavors")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"operator is missing required fields: {', '.join(missing)}")
    requested_mode = canonical_input(record.get("mode", mode))
    # The top-level input selects the representation. A per-record mode is
    # accepted only when it agrees, avoiding silently mixed correlators.
    if requested_mode != mode:
        raise ValueError("all operator modes must match top-level input")
    label = record.get("label", f"{side}_{record['species']}_{index}")
    flavors = record["flavors"]
    if not isinstance(flavors, list):
        flavors = list(flavors)
    metadata = record.get("contraction_metadata", record.get("metadata"))
    kwargs = {
        "external_tag": record.get("external_tag", ""),
        "contraction_metadata": metadata,
        "mode": mode,
        "momentum": record.get("momentum"),
        "projector": record.get("projector"),
        "clebsch": record.get("clebsch"),
        "phase": record.get("phase"),
        "external_spin": record.get("external_spin"),
    }
    return operator_model.make_operator(
        side, str(record["kind"]), str(record["species"]), str(label),
        flavors, record.get("gamma", "") or "", record.get("gamma_slots"),
        record.get("open_slot", operator_model._DEFAULT_OPEN_SLOT), **kwargs,
    )


def load_unified_input(path: str | Path | Mapping[str, Any], *,
                       input_type: str | None = None, mode: str | None = None,
                       output: str | None = None):
    """Load a unified JSON document and return ``(config, operators, matrices)``."""
    if isinstance(path, Mapping):
        document = dict(path)
        source_name = str(document.get("name", "correlator"))
    else:
        source = Path(path)
        document = json.loads(source.read_text(encoding="utf-8"))
        source_name = str(document.get("name", source.stem))
    if not isinstance(document, Mapping):
        raise ValueError("input document must be a JSON object")
    input_value = canonical_input(
        input_type or document.get("input", document.get("input_type", document.get("operator_mode", "distillation")))
    )
    mode_value = canonical_mode(mode or document.get("mode", "wick"))
    output_value = canonical_output(output or document.get("output", "independent"))
    if mode_value == "special" and input_value != "normal":
        raise ValueError("mode='special' is available only for normal input")
    if output_value == "count" and mode_value != "special":
        raise ValueError("output='count' is available only for special mode")
    if output_value == "block" and not (mode_value == "wick" and input_value == "normal"):
        raise ValueError("output='block' is available only for normal-wick input")
    sink_records, source_records = _operator_records(document)
    operators = tuple(
        _make_operator(record, "sink", input_value, index)
        for index, record in enumerate(sink_records)
    ) + tuple(
        _make_operator(record, "source", input_value, index)
        for index, record in enumerate(source_records)
    )
    matrices = {
        name: core._parse_matrix(value)
        for name, value in document.get("gamma_matrices", {}).items()
    }
    config = UnifiedConfig(
        input=input_value, mode=mode_value, output=output_value, name=source_name,
        wick_scope=document.get("wick_scope", "complete"),
        normal_ordered=document.get("normal_ordered", False),
        isospin_degenerate_perams=document.get("isospin_degenerate_perams", True),
        tolerance=float(document.get("tolerance", 1.0e-12)),
        batch_size=int(document.get("batch_size", 4096)),
    )
    return config, operators, matrices


def _json_value(value: Any):
    if hasattr(value, "endpoints") and hasattr(value, "coordinates"):
        return {"endpoints": [_json_value(item) for item in value.endpoints],
                "coordinates": int(value.coordinates)}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    if isinstance(value, complex):
        return [value.real, value.imag]
    return repr(value)


def _number(value: complex):
    value = complex(value)
    return [value.real, value.imag]


def _json_tree(value: Any):
    """Make nested fused keys/statistics JSON-safe, including tuple keys."""
    if isinstance(value, Mapping):
        return {str(_json_value(key)): _json_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_tree(item) for item in value]
    if isinstance(value, complex):
        return _number(value)
    if hasattr(value, "endpoints") and hasattr(value, "coordinates"):
        return _json_value(value)
    return value


def _to_colour_operator(operator):
    """Translate the normalized model into the incidence/S3 model's API."""
    from kernels.wick_colour_orbit.user_api import HadronOperator, make_operator as make_colour_operator
    if isinstance(operator, HadronOperator):
        return operator
    metadata = core.contraction_metadata_dict(operator.contraction_metadata)
    external_spin = metadata.get("external_spin")
    # Normal mode carries x/p endpoint metadata on the peram. ``point`` is a
    # stable, JSON-safe label and remains distinct when momentum/projector
    # metadata differs.
    point = operator.side
    point_metadata = {key: value for key, value in metadata.items()
                      if key != "external_spin"}
    if operator.mode == "normal" and (operator.external_tag or point_metadata):
        point = json.dumps({"tag": operator.external_tag, "metadata": point_metadata},
                           ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return make_colour_operator(
        operator.side, operator.kind, operator.species, operator.label,
        operator.flavors, operator.gamma, operator.gamma_slots, operator.open_slot,
        operator.external_tag, external_spin=external_spin, point=point,
    )


def _special_correlator(operators):
    sink = tuple(_to_colour_operator(op) for op in operators if op.side == "sink")
    source = tuple(_to_colour_operator(op) for op in operators if op.side == "source")
    return [sink, source]


def _cross_time_fast_path_unavailable(error: Exception) -> bool:
    """Recognize inputs that need the graph kernel for cross-time filtering."""
    message = str(error)
    return (
        "same number of fields" in message
        or "cross-time Wick pairing" in message
    )


def run_unified(config: UnifiedConfig, operators: Sequence[Any], *,
                gamma_matrices: Mapping[str, Any] | None = None,
                max_entries: int | None = None) -> dict[str, Any]:
    """Execute one normalized input/mode/output selection in memory.

    ``special`` returns the incidence/S3 fused coefficient dictionary. Files
    are written through :func:`write_unified`, which keeps the sibling JSON
    report schema for special outputs.
    """
    config = config if isinstance(config, UnifiedConfig) else UnifiedConfig(**config)
    if len(operators) == 2 and all(isinstance(side, (list, tuple)) for side in operators):
        operators = tuple(item for side in operators for item in side)
    else:
        operators = tuple(operators)
    if config.mode == "wick":
        with tempfile.TemporaryDirectory(prefix="wick-unified-") as directory:
            path = Path(directory) / f"{_safe_name(config.name)}.py"
            report = write_wick_output(
                split_operators(operators), path, name=config.name,
                mode=config.output, isospin_degenerate=config.isospin_degenerate_perams,
                gamma_matrices=gamma_matrices,
            )
        report.update({"input": config.input, "mode": config.mode, "output": config.output})
        return report

    from kernels.wick_colour_orbit import flatten_connectivity
    from kernels.wick_colour_orbit.cross_time import (
        accumulate_cross_time_direct,
        accumulate_cross_time_fused,
    )
    from kernels.wick_colour_orbit.general import accumulate_general_fused
    from kernels.wick_colour_orbit.user_api import (
        normalize_fast_correlator,
        normalize_general_fast_correlator,
    )
    if config.output == "count":
        # Count-only mode uses the bounded-memory exact SQLite merge. This
        # keeps ppnn feasible while retaining the ordinary-colour/S3 count.
        from kernels.wick_colour_orbit import count_colour_orbit_sqlite, normalize_correlator
        with tempfile.TemporaryDirectory(prefix="wick-special-count-") as directory:
            exact_correlator, _registry = normalize_correlator(
                _special_correlator(operators), gamma_matrices=gamma_matrices,
            )
            summary = count_colour_orbit_sqlite(
                exact_correlator, Path(directory) / "terms.sqlite3",
                normal_ordered=config.normal_ordered,
                wick_scope=config.wick_scope,
                isospin_degenerate_perams=config.isospin_degenerate_perams,
                batch_size=config.batch_size,
                zero_tolerance=config.tolerance,
            )
        return {"name": config.name, "input": config.input,
                "mode": config.mode, "output": config.output, **summary}

    colour_operators = _special_correlator(operators)
    fast = None
    generalized = None
    if config.wick_scope == "cross_time":
        # Cross-time inputs use the compact source-coefficient representation.
        # This is the retained specialized kernel, selected explicitly rather
        # than through a second automatic dispatcher.
        try:
            fast = normalize_fast_correlator(
                colour_operators, gamma_matrices=gamma_matrices,
            )
        except (TypeError, ValueError) as error:
            if not _cross_time_fast_path_unavailable(error):
                raise
            # A balanced graph may still have no all-crossing pairing (for
            # example, mismatched sink/source flavours). Keep the scope and
            # evaluate it with the general graph kernel instead of rejecting
            # the legal input or dispatching through an automatic router.
            generalized = normalize_general_fast_correlator(
                colour_operators, gamma_matrices=gamma_matrices,
            )
            values, stats = accumulate_general_fused(
                generalized,
                normal_ordered=config.normal_ordered,
                isospin_degenerate_perams=config.isospin_degenerate_perams,
                tolerance=config.tolerance,
                max_entries=max_entries,
                use_colour_orbits="auto",
                use_transporters=False,
                wick_scope=config.wick_scope,
                group_by_connectivity=True,
            )
            stats["route"] = "general-fused-cross-time"
            stats["fallback_reason"] = (
                "packed representation unavailable; used the general graph "
                "kernel with the cross_time filter"
            )
        else:
            values, stats = accumulate_cross_time_fused(
                fast, tolerance=config.tolerance, max_entries=max_entries,
            )
        coefficients = (
            dict(values) if fast is not None else flatten_connectivity(values)
        )
    else:
        generalized = normalize_general_fast_correlator(
            colour_operators, gamma_matrices=gamma_matrices,
        )
        values, stats = accumulate_general_fused(
            generalized,
            normal_ordered=config.normal_ordered,
            isospin_degenerate_perams=config.isospin_degenerate_perams,
            tolerance=config.tolerance,
            max_entries=max_entries,
            use_colour_orbits="auto",
            use_transporters=False,
            wick_scope=config.wick_scope,
            group_by_connectivity=True,
        )
        coefficients = flatten_connectivity(values)
    result = {"name": config.name, "input": config.input,
              "mode": config.mode, "output": config.output,
              "representation": "incidence-s3-packed",
              "nlist": len(coefficients), "stats": stats,
              "coefficients": coefficients}
    if config.output == "check":
        if max_entries is not None and len(coefficients) > max_entries:
            result["check"] = {"status": "skipped", "reason": "max_entries bound"}
        else:
            if config.wick_scope == "cross_time" and fast is not None:
                direct = accumulate_cross_time_direct(
                    fast, tolerance=config.tolerance, max_entries=max_entries,
                )
                oracle = "wick_colour_orbit.cross_time_direct"
            else:
                if generalized is None:
                    generalized = normalize_general_fast_correlator(
                        colour_operators, gamma_matrices=gamma_matrices,
                    )
                direct_values, _direct_stats = accumulate_general_fused(
                    generalized,
                    normal_ordered=config.normal_ordered,
                    isospin_degenerate_perams=config.isospin_degenerate_perams,
                    tolerance=config.tolerance,
                    max_entries=max_entries,
                    use_colour_orbits=False,
                    use_transporters=False,
                    wick_scope=config.wick_scope,
                    group_by_connectivity=True,
                )
                direct = flatten_connectivity(direct_values)
                oracle = "wick_colour_orbit.general_fused_without_s3"
            mismatches = sum(
                abs(complex(direct.get(key, 0)) - complex(coefficients.get(key, 0))) > config.tolerance
                for key in set(direct) | set(coefficients)
            )
            result["check"] = {
                "status": "passed" if mismatches == 0 else "failed",
                "oracle": oracle,
                "mismatches": mismatches,
                "oracle_entries": len(direct),
            }
    return result


def write_unified(config: UnifiedConfig, operators: Sequence[Any], *,
                  gamma_matrices: Mapping[str, Any] | None = None,
                  result_root: str | Path = "result",
                  max_entries: int | None = None) -> dict[str, Any]:
    """Run and persist under ``result/input_mode_output``.

    The established Wick Python and colour-orbit JSON writers are reused.
    This keeps downstream loaders compatible with both source projects.
    """
    if len(operators) == 2 and all(isinstance(side, (list, tuple)) for side in operators):
        operators = tuple(item for side in operators for item in side)
    directory = result_directory(config, result_root)
    suffix = ".py" if config.mode == "wick" else ".json"
    path = directory / f"{_safe_name(config.name)}_{config.input}_{config.mode}_{config.output}{suffix}"
    if config.mode == "wick":
        result = write_wick_output(
            split_operators(operators), path, name=config.name,
            mode=config.output, isospin_degenerate=config.isospin_degenerate_perams,
            gamma_matrices=gamma_matrices,
        )
    else:
        from kernels.wick_colour_orbit import write_count_output, write_output
        colour_ops = _special_correlator(operators)
        if config.output == "count":
            result = write_count_output(
                colour_ops, path, name=config.name,
                normal_ordered=config.normal_ordered, wick_scope=config.wick_scope,
                isospin_degenerate_perams=config.isospin_degenerate_perams,
                tolerance=config.tolerance, gamma_matrices=gamma_matrices,
                batch_size=config.batch_size,
            )
        else:
            result = write_output(
                colour_ops, path, name=config.name, mode=config.output,
                normal_ordered=config.normal_ordered, wick_scope=config.wick_scope,
                isospin_degenerate_perams=config.isospin_degenerate_perams,
                tolerance=config.tolerance, gamma_matrices=gamma_matrices,
            )
    result.update({"input": config.input, "mode": config.mode, "output": config.output,
                   "result_path": str(path), "result_directory": str(directory)})
    manifest = {
        "name": config.name, "input": config.input, "mode": config.mode,
        "output": config.output, "wick_scope": config.wick_scope,
        "operator_mode": config.input, "result_file": path.name,
        "source": "wick_reducer.unified",
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "--input", dest="input_file", required=True, type=Path)
    parser.add_argument("--input-type", "--representation", choices=("distillation", "distrilation", "normal"))
    parser.add_argument("--mode", choices=("wick", "special"))
    parser.add_argument("--output", choices=("independent", "indenpent", "check", "block", "count"))
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    parser.add_argument("--max-entries", type=int)
    args = parser.parse_args(argv)
    config, operators, matrices = load_unified_input(
        args.input_file, input_type=args.input_type, mode=args.mode, output=args.output,
    )
    result = write_unified(config, operators, gamma_matrices=matrices,
                           result_root=args.result_root, max_entries=args.max_entries)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"coefficients", "report", "plan", "block_plan"}},
                     ensure_ascii=False, default=_json_value))
    return 0


__all__ = [
    "UnifiedConfig", "canonical_input", "canonical_mode", "canonical_output",
    "load_unified_input", "result_directory", "run_unified", "write_unified", "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
