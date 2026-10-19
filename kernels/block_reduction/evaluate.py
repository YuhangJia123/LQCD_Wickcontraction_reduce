"""Numerically execute stand-alone symbolic expressions and W block plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Callable

from .model import BlockPlan, Network, TensorSpec, as_network, as_view


def _backend(contract):
    if contract is not None:
        return contract
    from opt_einsum import contract as default_contract
    return default_contract


def _checked_arrays(specs: Mapping[str, TensorSpec], arrays: Mapping):
    import numpy as np

    result = {}
    identities = {}

    def alias_layout(spec):
        return (spec.shape, spec.axis_types, spec.kind, spec.mode, spec.peram_ports, spec.symmetries)

    for name, spec in specs.items():
        if spec.mode != "normal":
            raise ValueError("evaluate_plan requires normal-mode TensorSpecs")
        if any(kind == "spin" and size != 4 for kind, size in zip(spec.axis_types, spec.shape)):
            raise ValueError("evaluate_plan requires all four Dirac components")
        if spec.kind == "peram" and any(
            spec.shape[color] != 3 or spec.shape[spin] != 4
            for color, spin in spec.peram_ports
        ):
            raise ValueError(
                "normal propagator ports require (color, spin)=(3, 4) at both ends"
            )
        if spec.kind == "wedge":
            continue
        if name in arrays:
            value = np.asarray(arrays[name])
        elif spec.identity is not None and spec.identity in arrays:
            value = np.asarray(arrays[spec.identity])
        else:
            continue
        if tuple(value.shape) != tuple(spec.shape):
            raise ValueError(f"tensor {name!r} has shape {value.shape}, expected {spec.shape}")
        identity = spec.identity if spec.identity is not None else name
        if identity in identities:
            previous_name, previous_spec, previous_value = identities[identity]
            if alias_layout(previous_spec) != alias_layout(spec):
                raise ValueError(f"tensor aliases {previous_name!r} and {name!r} have incompatible slot layouts or symmetries")
            if not np.array_equal(previous_value, value):
                raise ValueError(f"tensor aliases {previous_name!r} and {name!r} must contain the same values")
        else:
            identities[identity] = (name, spec, value)
        result[name] = value
    for name, spec in specs.items():
        if name in result or spec.kind == "wedge":
            continue
        identity = spec.identity if spec.identity is not None else name
        if identity not in identities:
            continue
        previous_name, previous_spec, value = identities[identity]
        if alias_layout(previous_spec) != alias_layout(spec):
            raise ValueError(f"tensor aliases {previous_name!r} and {name!r} have incompatible slot layouts or symmetries")
        result[name] = value
    return result


@lru_cache(maxsize=256)
def _contraction_path(inputs, output, shapes):
    """Cache only shape/topology metadata, never numeric tensor arrays."""
    from opt_einsum import contract_path, get_symbol

    # The shape-only interleaved API is not supported by all opt_einsum
    # versions. Its equation API also supports arbitrary Unicode labels.
    equation = ",".join("".join(get_symbol(index) for index in ports) for ports in inputs)
    equation += "->" + "".join(get_symbol(index) for index in output)
    path, _ = contract_path(equation, *shapes, shapes=True, optimize="greedy")
    return tuple(tuple(step) for step in path)


def _execute(network: Network, arrays: Mapping, contract: Callable, *, cache_path=False):
    """Use integer einsum labels, so graph indices are not limited to letters."""
    import numpy as np

    labels = {}
    operands = []
    inputs = []
    shapes = []
    dimensions = {}
    for factor in network.factors:
        if factor.tensor not in arrays:
            raise KeyError(f"missing tensor array {factor.tensor!r}")
        value = arrays[factor.tensor]
        if len(factor.indices) != value.ndim:
            raise ValueError(f"tensor {factor.tensor!r} rank does not match its network ports")
        for label, dimension in zip(factor.indices, value.shape):
            if label in dimensions and dimensions[label] != dimension:
                raise ValueError(f"network index {label!r} has incompatible dimensions")
            dimensions[label] = dimension
        ports = [labels.setdefault(label, len(labels)) for label in factor.indices]
        operands.extend((value, ports))
        inputs.append(tuple(ports))
        shapes.append(tuple(value.shape))
    if not network.factors:
        if network.output:
            raise ValueError("a network without factors cannot have output indices")
        return np.asarray(1.0)
    if len(set(network.output)) != len(network.output) or any(label not in labels for label in network.output):
        raise ValueError("network output indices must be distinct existing ports")
    output = tuple(labels[label] for label in network.output)
    operands.append(list(output))
    optimize = _contraction_path(tuple(inputs), output, tuple(shapes)) if cache_path else "greedy"
    return np.asarray(contract(*operands, optimize=optimize))


def _permutation(permutation, rank: int):
    value = tuple(permutation) if permutation else tuple(range(rank))
    if sorted(value) != list(range(rank)):
        raise ValueError("array permutation must contain each output axis exactly once")
    return value


class _Accumulator:
    """Keep one sum and one reusable workspace, regardless of view count."""

    def __init__(self):
        self.value = None
        self.workspace = None

    def add(self, value, coefficient):
        import numpy as np

        if self.value is None:
            self.value = np.asarray(np.multiply(value, coefficient))
            return
        if self.value.shape != value.shape:
            raise ValueError(f"target views have incompatible shapes {self.value.shape} and {value.shape}")
        dtype = np.result_type(self.value.dtype, np.result_type(value, coefficient))
        if self.value.dtype != dtype:
            self.value = self.value.astype(dtype)
        if self.workspace is None or self.workspace.dtype != dtype:
            self.workspace = np.empty(value.shape, dtype=dtype)
        np.multiply(value, coefficient, out=self.workspace)
        np.add(self.value, self.workspace, out=self.value)

    def finish(self, zero_shape=()):
        import numpy as np

        return np.zeros(zero_shape, dtype=np.complex128) if self.value is None else self.value


def evaluate_expression(contracts: Mapping, basis: Mapping | Sequence, coefficients: Mapping | Sequence | None, arrays: Mapping, contract=None):
    """Evaluate each referenced contract once and immediately sum its views.

    ``contracts`` contains Network values; ``basis`` contains BasisView values.
    Target coefficients may be keyed by arbitrary basis keys.  Passing None
    uses the coefficient embedded in each BasisView, with external weight one.
    Array shapes for each repeated index and all target output shapes are
    validated.  Tensor-spec validation belongs to the symbolic input/planner;
    evaluate_plan additionally validates every supplied array against its spec.
    Numeric storage does not grow with the number of contracts or views.
    Grouping views by contract can change floating-point addition order.
    """
    import numpy as np

    backend = _backend(contract)
    prepared = {name: np.asarray(value) for name, value in arrays.items()}
    views = basis.items() if isinstance(basis, Mapping) else enumerate(basis)
    grouped = {}
    for key, view in views:
        view = as_view(view)
        try:
            coefficient = 1 if coefficients is None else coefficients[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise KeyError(f"missing or invalid target coefficient for basis view {key!r}") from exc
        if view.contract_id not in contracts:
            raise ValueError(f"basis references unknown contract {view.contract_id!r}")
        coefficient = complex(coefficient) * complex(view.coefficient) * complex(view.scalar)
        grouped.setdefault(view.contract_id, []).append((coefficient, view.permutation))
    accumulator = _Accumulator()
    for index, network in contracts.items():
        group = grouped.get(index)
        if not group:
            _execute(as_network(network), prepared, backend, cache_path=contract is None)
            continue
        value = _execute(as_network(network), prepared, backend, cache_path=contract is None)
        for coefficient, axes in group:
            permutation = _permutation(axes, value.ndim)
            accumulator.add(value.transpose(permutation), coefficient)
        del value
    return accumulator.finish()


def evaluate_plan(plan: BlockPlan, arrays: Mapping, contract=None):
    """Build every shared full-spin W once, then evaluate each planned network.

    W ports are ordered ``left1 color/spin, left2 color/spin,
    right1 color/spin, right2 color/spin``.  ``BlockDefinition.ports`` gives
    the propagator's ordered left and right joint-axis pairs, independently
    of the original array layout.  No spin projection or truncation is done.
    Each used network is computed once, its signed transpose views are added
    immediately, and its array is released before the next contraction. The
    default backend reuses a bounded, shape-only contraction-path cache.
    """
    import numpy as np

    backend = _backend(contract)
    prepared = _checked_arrays(plan.tensor_specs, arrays)
    built = set()
    for definition in plan.w_definitions:
        if definition.name in built:
            raise ValueError(f"duplicate W definition {definition.name!r}")
        if definition.propagator not in prepared:
            raise KeyError(f"missing W propagator {definition.propagator!r}")
        propagator_spec = plan.tensor_specs.get(definition.propagator)
        if propagator_spec is None or propagator_spec.kind != "peram" or propagator_spec.mode != "normal":
            raise ValueError("W definitions require an explicitly declared normal-mode propagator")
        propagator = prepared[definition.propagator]
        ports = tuple(tuple(pair) for pair in definition.ports)
        if len(ports) != 2 or any(len(pair) != 2 for pair in ports):
            raise ValueError("W ports must give two joint color/spin axis pairs")
        order = tuple(axis for pair in ports for axis in pair)
        if propagator.ndim != 4 or sorted(order) != list(range(4)):
            raise ValueError("W propagators require four distinct full color/spin axes")
        if ports != propagator_spec.peram_ports:
            raise ValueError("W definition ports disagree with the propagator TensorSpec")
        ordered = propagator.transpose(order)
        left_shape, right_shape = ordered.shape[:2], ordered.shape[2:]
        if left_shape != (3, 4) or right_shape != (3, 4):
            raise ValueError(
                "W requires color dimension 3 and all four Dirac components at both ends"
            )
        matrix = ordered.reshape(12, 12)
        value = matrix[:, None, :, None] * matrix[None, :, None, :]
        np.subtract(value, matrix[:, None, None, :] * matrix[None, :, :, None], out=value)
        shape = (*left_shape, *left_shape, *right_shape, *right_shape)
        if tuple(definition.shape) != tuple(shape):
            raise ValueError(f"W {definition.name!r} declares shape {definition.shape}, expected {shape}")
        value = value.reshape(shape)
        if definition.name in plan.tensor_specs:
            spec = plan.tensor_specs[definition.name]
            if spec.kind != "wedge" or tuple(spec.shape) != shape or spec.axis_types != ("color", "spin") * 4:
                raise ValueError(f"W {definition.name!r} TensorSpec disagrees with its definition")
        prepared[definition.name] = value
        built.add(definition.name)
    grouped = {}
    zero_shape = tuple(plan.stats.get("output_shape", ()))
    for term in plan.reconstruction:
        if not isinstance(term.network_id, int) or not 0 <= term.network_id < len(plan.networks):
            raise ValueError(f"reconstruction references unknown network {term.network_id!r}")
        grouped.setdefault(term.network_id, []).append(term)
    accumulator = _Accumulator()
    for index, network in enumerate(plan.networks):
        group = grouped.get(index)
        if not group:
            _execute(network, prepared, backend, cache_path=contract is None)
            continue
        value = _execute(network, prepared, backend, cache_path=contract is None)
        for term in group:
            permutation = _permutation(term.permutation, value.ndim)
            accumulator.add(value.transpose(permutation), complex(term.coefficient))
        del value
    if not grouped and not zero_shape and plan.original_terms:
        original = plan.original_terms[0].network
        dimensions = {}
        for factor in original.factors:
            for label, size in zip(factor.indices, prepared[factor.tensor].shape):
                dimensions[label] = size
        zero_shape = tuple(dimensions[label] for label in original.output)
    return accumulator.finish(zero_shape)


__all__ = ["evaluate_expression", "evaluate_plan"]
