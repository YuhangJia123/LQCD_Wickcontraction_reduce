"""Explicit multi-transporter Wick reduction for qct distillation graphs."""

from .reducer import (
    ReductionOptions, benchmark_modes, main, reduce_operators,
    run_examples,
)
from .operator_model import make_operator, operator_library
from .expression_output import (
    contraction_expression, normalize_correlator, render_python_output,
    default_block_output_path, split_operators, write_case_output, write_output,
)

__all__ = [
    "ReductionOptions", "benchmark_modes", "main", "make_operator", "operator_library",
    "UnifiedConfig", "canonical_input", "canonical_mode", "canonical_output",
    "load_unified_input", "result_directory", "run_unified", "write_unified",
    "reduce_operators", "run_examples",
    "contraction_expression", "normalize_correlator", "render_python_output",
    "default_block_output_path", "split_operators", "write_case_output", "write_output",
]


_UNIFIED_EXPORTS = {
    "UnifiedConfig", "canonical_input", "canonical_mode", "canonical_output",
    "load_unified_input", "result_directory", "run_unified",
    "write_unified",
}

def __getattr__(name):
    if name in _UNIFIED_EXPORTS:
        from . import unified
        value = getattr(unified, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
