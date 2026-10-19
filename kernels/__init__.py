"""Production calculation kernels with lazily loaded convenience exports.

Importing the block planner does not load the Wick reducer or its operator
model. The unified command-line entry point is ``python -m kernels``.
"""

__all__ = [
    "benchmark_modes",
    "ReductionOptions",
    "contraction_expression",
    "make_operator",
    "normalize_correlator",
    "operator_library",
    "reduce_operators",
    "render_python_output",
    "run_examples",
    "split_operators",
    "write_case_output",
    "write_output",
]


def __getattr__(name):
    if name in __all__:
        from . import wick_reducer

        value = getattr(wick_reducer, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
