"""Standalone ordinary-colour Wick coefficient reduction."""

from .model import (
    Correlator,
    EndpointResolvedKey,
    EndpointRoute,
    Field as FastField,
    LocalVertex,
    Operator,
    FullWick,
    WickMap,
    enumerate_general_wicks,
    enumerate_cross_time_wicks,
)
from .cross_time import accumulate_cross_time_direct, accumulate_cross_time_fused
from .general import accumulate_general_fused
from .fusion import decode_digits, encode_digits
from .grouping import GroupedWickResult, flatten_connectivity
from .user_api import (
    adapt_operator,
    make_operator,
    normalize_correlator,
    normalize_fast_correlator,
    normalize_general_fast_correlator,
)
from .input_json import INPUT_FORMAT, load_input, parse_input_document


# Sparse file writers are loaded lazily so the two in-memory fusion kernels
# remain independent from the SQLite and JSON output backends.
from importlib import import_module as _import_module

_SPARSE_EXPORTS = {
    "SparseCorrelator": "Correlator",
    "Field": "Field",
    "SparseOperator": "Operator",
    "TensorTerm": "TensorTerm",
    "count_colour_orbit_sqlite": "count_colour_orbit_sqlite",
    "count_exact": "count_exact",
    "count_exact_sqlite": "count_exact_sqlite",
    "iter_grouped_count_representatives_sqlite": "iter_grouped_count_representatives_sqlite",
    "epsilon_three_quark_operator": "epsilon_three_quark_operator",
    "explicit_operator": "explicit_operator",
    "meson_operator": "meson_operator",
    "product_operator": "product_operator",
}


def __getattr__(name):
    if name in _SPARSE_EXPORTS:
        value = getattr(_import_module(".exact_operator_backend", __name__),
                        _SPARSE_EXPORTS[name])
    elif name in {"write_output", "write_count_output"}:
        value = getattr(_import_module(".output", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "Correlator",
    "EndpointResolvedKey",
    "EndpointRoute",
    "Field",
    "FastField",
    "LocalVertex",
    "Operator",
    "SparseCorrelator",
    "SparseOperator",
    "TensorTerm",
    "count_colour_orbit_sqlite",
    "count_exact",
    "count_exact_sqlite",
    "iter_grouped_count_representatives_sqlite",
    "WickMap",
    "FullWick",
    "accumulate_cross_time_direct",
    "accumulate_cross_time_fused",
    "accumulate_general_fused",
    "GroupedWickResult",
    "flatten_connectivity",
    "decode_digits",
    "encode_digits",
    "enumerate_cross_time_wicks",
    "enumerate_general_wicks",
    "make_operator",
    "adapt_operator",
    "normalize_correlator",
    "normalize_fast_correlator",
    "normalize_general_fast_correlator",
    "explicit_operator",
    "product_operator",
    "meson_operator",
    "epsilon_three_quark_operator",
    "write_output",
    "write_count_output",
    "INPUT_FORMAT",
    "load_input",
    "parse_input_document",
]
