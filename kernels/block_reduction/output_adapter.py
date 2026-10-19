"""Project-facing normal-output adapter for normal-wick-block.

Importing this module never imports the reducer.  Only
``project_gamma_arrays`` asks the project's matrix registry for constants.
The symbolic reader treats the generated source as data; it does not execute it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import itertools
import json
from pathlib import Path
import re
import string
from types import MappingProxyType
from typing import Mapping

import numpy as np

from kernels.block_reduction import BasisView, Factor, Network, TensorSpec, TensorSymmetry, WeightedNetwork
from kernels.block_reduction.io import SymbolicInput, load_symbolic_output


_PERAM = re.compile(r"^peram_(.+?)_(sin|sou)_(sin|sou)_x_x(?:_p_([A-Za-z0-9_]+))?$")
_ENDPOINT_COMMENT = re.compile(r"^#\s+(peram_[A-Za-z0-9_]+):\s+(\{.*\})\s*$")


@dataclass(frozen=True)
class PeramMetadata:
    flavor: str
    quark_side: str
    antiquark_side: str
    endpoint_tag: str | None
    endpoints: Mapping[str, object] | None


@dataclass(frozen=True)
class MatrixRelation:
    """A[source] == sign * A[target].transpose(permutation)."""

    source: str
    target: str
    permutation: tuple[int, int]
    sign: int


@dataclass(frozen=True)
class AdaptedNormalOutput:
    symbolic: SymbolicInput
    constant_arrays: Mapping[str, np.ndarray]
    peram_metadata: Mapping[str, PeramMetadata]
    matrix_relations: tuple[MatrixRelation, ...]
    original_symbolic: SymbolicInput

    @property
    def basis_contract(self):
        return self.symbolic.basis_contract

    @property
    def basis(self):
        return self.symbolic.basis

    @property
    def coefficients(self):
        return self.symbolic.coefficients

    @property
    def tensor_specs(self):
        return self.symbolic.tensor_specs


def project_gamma_arrays() -> dict[str, np.ndarray]:
    """Read the registered project Gamma matrices outside the block core."""
    from kernels import correlator_wick_normalizer as core
    from kernels.wick_reducer.expression_output import _gamma_symbol

    return {_gamma_symbol(name): np.asarray(matrix, dtype=np.complex128)
            for name, matrix in core._euclidean_gamma_matrices().items()}


def _epsilon_color() -> np.ndarray:
    epsilon = np.zeros((3, 3, 3), dtype=np.int8)
    for permutation in itertools.permutations(range(3)):
        inversions = sum(permutation[i] > permutation[j]
                         for i in range(3) for j in range(i + 1, 3))
        epsilon[permutation] = (-1) ** inversions
    return epsilon


def _source_text(source_or_path: str | Path) -> str:
    if isinstance(source_or_path, Path):
        return source_or_path.read_text(encoding="utf-8")
    if not isinstance(source_or_path, str):
        raise TypeError("source_or_path must be source text or a path")
    if "\n" not in source_or_path and "\r" not in source_or_path:
        try:
            path = Path(source_or_path)
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            pass
    return source_or_path


def _operand_names(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "contract"):
            for operand in node.args[1:]:
                if not isinstance(operand, ast.Name):
                    raise ValueError("normal output requires named tensor operands")
                names.add(operand.id)
    return names


def _endpoint_comments(source: str) -> dict[str, Mapping[str, object]]:
    metadata = {}
    for line in source.splitlines():
        match = _ENDPOINT_COMMENT.fullmatch(line)
        if not match:
            continue
        try:
            endpoints = json.loads(match.group(2))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid endpoint metadata for {match.group(1)}") from error
        if not isinstance(endpoints, dict) or set(endpoints) != {"quark", "antiquark"}:
            raise ValueError(f"invalid endpoint metadata for {match.group(1)}")
        if match.group(1) in metadata and metadata[match.group(1)] != endpoints:
            raise ValueError(f"conflicting endpoint metadata for {match.group(1)}")
        metadata[match.group(1)] = MappingProxyType(endpoints)
    return metadata


def _alias_identity(name: str, aliases: Mapping[str, str], names: set[str]) -> str:
    seen = set()
    while name in aliases:
        if name in seen:
            raise ValueError("cyclic tensor alias declarations")
        seen.add(name)
        name = aliases[name]
        if name not in names:
            raise ValueError(f"tensor alias target {name!r} is absent from this output")
    return name


def _matrix_symmetries(array: np.ndarray) -> tuple[TensorSymmetry, ...]:
    if np.array_equal(array.T, -array):
        return (TensorSymmetry((1, 0), -1),)
    if np.array_equal(array.T, array):
        return (TensorSymmetry((1, 0), 1),)
    return ()


def _relations(arrays: Mapping[str, np.ndarray]) -> tuple[MatrixRelation, ...]:
    result = []
    for source in sorted(arrays):
        for target in sorted(arrays):
            for permutation in ((0, 1), (1, 0)):
                for sign in (1, -1):
                    if np.array_equal(arrays[source], sign * arrays[target].transpose(permutation)):
                        result.append(MatrixRelation(source, target, permutation, sign))
    return tuple(result)


def _normalize_matrices(symbolic: SymbolicInput, relations: tuple[MatrixRelation, ...],
                        specs: Mapping[str, TensorSpec]) -> SymbolicInput:
    """Optionally fold exact registered matrix aliases into the three tables."""
    by_source: dict[str, list[MatrixRelation]] = {}
    for relation in relations:
        by_source.setdefault(relation.source, []).append(relation)
    chosen = {}
    for name, options in by_source.items():
        chosen[name] = min(options, key=lambda r: (r.target, r.permutation != (0, 1), r.sign < 0))
    contracts = {}
    signs = {}
    for contract_id, network in symbolic.basis_contract.items():
        factors = []
        scalar = 1
        for factor in network.factors:
            relation = chosen.get(factor.tensor)
            if relation is None:
                factors.append(factor)
                continue
            factors.append(Factor(relation.target, tuple(factor.indices[i]
                                                         for i in relation.permutation)))
            scalar *= relation.sign
        contracts[contract_id] = Network(tuple(factors), network.output)
        signs[contract_id] = scalar
    basis = {view_id: BasisView(view.contract_id, view.scalar * signs[view.contract_id],
                                view.permutation, view.coefficient)
             for view_id, view in symbolic.basis.items()}
    return SymbolicInput(MappingProxyType(contracts), MappingProxyType(basis),
                         symbolic.coefficients, MappingProxyType(dict(specs)), symbolic.prefix)


def load_normal_output(source_or_path: str | Path, *,
                       gamma_arrays: Mapping[str, np.ndarray] | None = None,
                       aliases: Mapping[str, str] | None = None,
                       extra_specs: Mapping[str, TensorSpec] | None = None,
                       extra_arrays: Mapping[str, np.ndarray] | None = None,
                       normalize_matrices: bool = False) -> AdaptedNormalOutput:
    """Adapt generated normal tables without executing source or changing them.

    Identical names denote one numerical array. Different names remain distinct
    unless ``aliases`` explicitly equates them. Unknown operands require an
    explicit spec; supplied constant symmetries are checked on their arrays.
    ``normalize_matrices`` is optional because a separate raw Wick family must
    undergo the same transformation before a target mapping can be certified.
    """
    source = _source_text(source_or_path)
    if "# Operator mode: normal." not in source:
        raise ValueError("the adapter requires an explicit normal-mode export")
    names = _operand_names(source)
    aliases = dict(aliases or {})
    if any(name not in names for name in aliases):
        raise ValueError("tensor alias source is absent from this output")
    known = dict(gamma_arrays if gamma_arrays is not None else project_gamma_arrays())
    known.update(extra_arrays or {})
    for name, array in (("epsilon_color", _epsilon_color()), ("delta_color", np.eye(3))):
        if name in known and not np.array_equal(known[name], array):
            raise ValueError(f"registered normal color tensor {name!r} is incorrect")
        known[name] = array
    comments = _endpoint_comments(source)
    supplied = dict(extra_specs or {})
    specs = {}
    metadata = {}
    for name in sorted(names):
        identity = _alias_identity(name, aliases, names)
        if name.startswith("peram_"):
            match = _PERAM.fullmatch(name)
            if match is None:
                raise ValueError(f"normal peram name lacks direction/endpoints: {name!r}")
            flavor, quark_side, antiquark_side, endpoint_tag = match.groups()
            if endpoint_tag and name not in comments:
                raise ValueError(f"missing endpoint metadata for {name!r}")
            metadata[name] = PeramMetadata(flavor, quark_side, antiquark_side,
                                           endpoint_tag, comments.get(name))
            specs[name] = TensorSpec(name, (3, 4, 3, 4),
                                     ("color", "spin", "color", "spin"),
                                     kind="peram", identity=identity,
                                     peram_ports=((0, 1), (2, 3)))
        elif name == "epsilon_color":
            specs[name] = TensorSpec(name, (3, 3, 3), ("color",) * 3,
                                     symmetries=(TensorSymmetry((1, 0, 2), -1),
                                                 TensorSymmetry((1, 2, 0), 1)))
        elif name == "delta_color":
            specs[name] = TensorSpec(name, (3, 3), ("color",) * 2,
                                     symmetries=(TensorSymmetry((1, 0), 1),))
        elif name in supplied:
            specs[name] = replace(supplied[name], name=name, identity=identity)
        elif name.startswith("W_"):
            # ``normal_wick_block`` documents every W as a rank-8 tensor built from
            # two complete normal perams.  The block definition itself is
            # executable Python, but the static symbolic reader must know its
            # ports before parsing basis_contract entries that use W factors.
            specs[name] = TensorSpec(
                name, (3, 4) * 4, ("color", "spin") * 4,
                kind="wedge", identity=identity,
            )
        elif name in known:
            array = np.asarray(known[name])
            if array.shape != (4, 4):
                raise ValueError(f"registered Gamma {name!r} is not 4x4")
            specs[name] = TensorSpec(name, (4, 4), ("spin", "spin"),
                                     identity=identity, symmetries=_matrix_symmetries(array))
        else:
            raise ValueError(f"unknown tensor {name!r} needs an explicit TensorSpec")
    for name, spec in specs.items():
        if name in known:
            array = np.asarray(known[name])
            if tuple(array.shape) != spec.shape:
                raise ValueError(f"constant array shape differs from TensorSpec for {name!r}")
            for symmetry in spec.symmetries:
                if not np.array_equal(array.transpose(symmetry.permutation), symmetry.sign * array):
                    raise ValueError(f"unproved tensor symmetry for {name!r}")
        elif spec.symmetries:
            raise ValueError(f"constant array required to verify symmetries for {name!r}")
        if spec.mode != "normal":
            raise ValueError(f"tensor {name!r} is not declared normal-mode")
        if spec.kind == "peram" and (spec.shape != (3, 4, 3, 4)
                                       or spec.axis_types != ("color", "spin", "color", "spin")
                                       or spec.peram_ports != ((0, 1), (2, 3))
                                       or spec.symmetries):
            raise ValueError("normal perams require four Dirac components and no inferred transpose symmetry")
        identity = _alias_identity(name, aliases, names)
        if identity != name:
            target = specs[identity]
            if (spec.shape, spec.axis_types, spec.kind, spec.peram_ports, spec.symmetries) != (
                    target.shape, target.axis_types, target.kind, target.peram_ports, target.symmetries):
                raise ValueError(f"tensor alias {name!r} has incompatible port definitions")
            if name in known and identity in known and not np.array_equal(known[name], known[identity]):
                raise ValueError(f"tensor alias {name!r} contradicts the registered arrays")
    original = load_symbolic_output(source, tensor_specs=specs)
    matrix_arrays = {name: np.asarray(array) for name, array in known.items()
                     if name in specs and specs[name].shape == (4, 4)}
    relations = _relations(matrix_arrays)
    adapted = _normalize_matrices(original, relations, specs) if normalize_matrices else original
    constants = {name: np.asarray(array) for name, array in known.items()
                 if name in specs and specs[name].kind != "peram"}
    return AdaptedNormalOutput(adapted, MappingProxyType(constants),
                               MappingProxyType(metadata), relations, original)


def pairing_family_from_operators(operators, *, isospin_degenerate=True,
                                  tensor_specs=None, compact=False):
    """Export optional raw pairings without putting project objects in core.

    The block planner must independently certify this family's total against
    the required three tables before accepting a plan based on it. This
    adapter does not claim that the raw and Gram-reduced sums are equivalent.
    """
    from kernels.wick_reducer import expression_output, operator_model
    from kernels.block_reduction.global_layout import CompactPairingTerms, PairingFamily

    operators = tuple(operators)
    if operator_model.core.validate_operator_modes(operators) != "normal":
        raise ValueError("raw pairing adapter requires normal operators")
    fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    terms = operator_model.core.enumerate_wicks(quarks, antiquarks)
    first = next(terms, None)
    if first is None:
        raise ValueError("the operators have no raw Wick pairing family")
    weight = expression_output._operator_scalar_weight(operators)
    if compact:
        if tensor_specs is None:
            raise ValueError("compact pairing family requires tensor_specs to validate every propagator")
        equation, operands, _output = expression_output._contraction_expression_from_fields(
            operators, fields, quarks, antiquarks, first.pairing,
            isospin_degenerate=isospin_degenerate,
        )
        positions = tuple(i for i, name in enumerate(operands)
                          if name.startswith("peram_"))
        if len(positions) != len(quarks):
            raise ValueError("raw pairings do not have one propagator per quark")
        classes_by_flavor = {}
        for slot, quark in enumerate(quarks):
            classes_by_flavor.setdefault(quark.flavor, []).append(slot)
        classes = tuple(tuple(slots) for _flavor, slots in sorted(classes_by_flavor.items()))
        table = []
        for sink, quark in enumerate(quarks):
            row = []
            for source in range(len(quarks)):
                antiquark = antiquarks[first.pairing[source]]
                row.append(expression_output._peram_symbol(
                    quark, antiquark, isospin_degenerate=isospin_degenerate,
                ) if quark.flavor == antiquark.flavor else None)
            table.append(tuple(row))
        reference = WeightedNetwork(Network.from_einsum(equation, operands), first.sign * weight)
        compact_terms = CompactPairingTerms(reference, positions, classes, tensor_specs,
                                           tuple(table))
        return PairingFamily(compact_terms, positions)

    networks = []
    positions = None
    from itertools import chain
    for term in chain((first,), terms):
        equation, operands, _output = expression_output._contraction_expression_from_fields(
            operators, fields, quarks, antiquarks, term.pairing,
            isospin_degenerate=isospin_degenerate,
        )
        row_positions = tuple(i for i, name in enumerate(operands)
                              if name.startswith("peram_"))
        if positions is None:
            positions = row_positions
        elif row_positions != positions:
            raise ValueError("propagator factor positions vary across raw pairings")
        if tensor_specs is not None and any(name not in tensor_specs for name in operands):
            missing = sorted(set(operands) - set(tensor_specs))
            raise ValueError(f"raw pairing uses tensors absent from the target specs: {missing}")
        networks.append(WeightedNetwork(Network.from_einsum(equation, operands),
                                        term.sign * weight))
    if len(positions) != len(quarks):
        raise ValueError("raw pairings do not have one propagator per quark")
    return PairingFamily(tuple(networks), positions)


def _py_identifier(value: str) -> str:
    """Render a tensor name as a safe Python identifier.

    Normal exports already use identifiers, but plans may be assembled by an
    external caller with names such as ``W:u``.  The symbolic plan keeps the
    original names; the output document uses the deterministic escaped name.
    """
    result = re.sub(r"\W", "_", str(value), flags=re.UNICODE)
    if not result or result[0].isdigit():
        result = "x_" + result
    return result


def _number_literal(value) -> str:
    """Keep block coefficients readable while preserving complex values."""
    value = complex(value)
    if value.imag == 0:
        real = float(value.real)
        return str(int(real)) if real.is_integer() else repr(real)
    return f"complex({value.real!r}, {value.imag!r})"


def _source_header(source_or_path) -> list[str]:
    """Return only the import/declaration header of an existing output file."""
    if source_or_path is None:
        return [
            "# Auto-generated block output.",
            "# Operator mode: normal. All tensor arrays are supplied by the caller.",
            "from opt_einsum import contract",
        ]
    source = _source_text(source_or_path)
    lines = source.splitlines()
    # Everything after this marker is an old expression table and must not be
    # copied into the block document.  Keep comments and tensor declarations so
    # the result remains directly executable with the same caller namespace.
    stop = len(lines)
    for index, line in enumerate(lines):
        if (line.startswith("# 最终系数数组")
                or line.startswith("# Block definitions:")
                or re.match(r"^[A-Za-z_]\w*_basis_contract\s*=", line)):
            stop = index
            break
    header = lines[:stop]
    if not any(line.strip() == "from opt_einsum import contract" for line in header):
        header.extend(("from opt_einsum import contract",))
    if not any("Operator mode: normal." in line for line in header):
        header.insert(1 if header else 0,
                     "# Operator mode: normal. All tensor arrays are supplied by the caller.")
    return header


def _network_call(network, specs: Mapping[str, object], names: Mapping[str, str]) -> str:
    """Convert a symbolic Network into a compact executable contract call."""
    labels = {}
    alphabet = string.ascii_letters

    def label(index):
        if index not in labels:
            position = len(labels)
            if position >= len(alphabet):
                raise ValueError("block output has more than 52 distinct einsum indices")
            labels[index] = alphabet[position]
        return labels[index]

    inputs = ["".join(label(index) for index in factor.indices)
              for factor in network.factors]
    output = "".join(label(index) for index in network.output)
    equation = ",".join(inputs) + "->" + output
    operands = [names.get(factor.tensor, _py_identifier(factor.tensor))
                for factor in network.factors]
    return f"contract({equation!r}, {', '.join(operands)})"


def _block_expression(definition, names: Mapping[str, str]) -> tuple[str, str, str]:
    """Build the two ordinary contractions in ``W = S S - S S``.

    A W tensor exposes four joint ports in the order
    ``left1, left2, right1, right2``.  ``BlockDefinition.ports`` maps each
    joint port back to the source perambulator axis positions, so this remains
    correct for any declared full-spin normal peram layout.
    """
    labels = tuple(string.ascii_lowercase[:8])
    left1, left2, right1, right2 = labels[:2], labels[2:4], labels[4:6], labels[6:8]

    def slots(left, right):
        slots = [None] * 4
        for axis, value in zip(definition.ports[0], left):
            slots[axis] = value
        for axis, value in zip(definition.ports[1], right):
            slots[axis] = value
        if any(value is None for value in slots):
            raise ValueError(f"invalid W ports for {definition.name!r}")
        return "".join(slots)

    # The output axis order is the same order used by evaluate_plan when it
    # reshapes the 12x12 propagator into two left and two right joint ports.
    propagator = names[definition.propagator]
    output = "".join(labels)
    first_equation = slots(left1, right1) + "," + slots(left2, right2) + "->" + output
    second_equation = slots(left1, right2) + "," + slots(left2, right1) + "->" + output
    first = f"contract({first_equation!r}, {propagator}, {propagator})"
    second = f"contract({second_equation!r}, {propagator}, {propagator})"
    return first, second, names[definition.name]


def render_block_output(plan, *, name: str = "normal_wick_block", source=None) -> str:
    """Render a :class:`~block_reduction.model.BlockPlan` as a normal output.

    The generated file has the same three expression tables as an independent
    output.  W tensors are defined first from the supplied full four-component
    perambulator arrays; each ``basis_contract`` then uses those W names as
    ordinary einsum operands.  Reconstruction terms provide the basis views,
    transposes, and coefficients, so no planner internals are required at
    execution time.
    """
    if not hasattr(plan, "w_definitions") or not hasattr(plan, "networks"):
        raise TypeError("plan must be a BlockPlan-like object")
    safe_name = _py_identifier(name)
    specs = dict(getattr(plan, "tensor_specs", {}))
    names = {tensor: _py_identifier(tensor) for tensor in specs}
    for definition in plan.w_definitions:
        names.setdefault(definition.propagator, _py_identifier(definition.propagator))
        names.setdefault(definition.name, _py_identifier(definition.name))

    lines = _source_header(source)
    lines.extend(("", "# Block definitions: full-spin antisymmetric two-propagator tensors."))
    if not plan.w_definitions:
        lines.append("# No W block was selected; normal-wick-block retains ordinary tensors only.")
    for definition in plan.w_definitions:
        first, second, block_name = _block_expression(definition, names)
        lines.extend((
            f"# {block_name} = S*S - S*S; ports={tuple(tuple(p) for p in definition.ports)!r}.",
            f"{block_name} = {first} - {second}",
        ))

    # A network may be referenced by several reconstruction views.  Emit one
    # contract per unique network, then let basis views carry only transposes.
    contract_ids = {}
    lines.extend(("", "# normal-wick-block contracts; keys start at 1.",
                  f"{safe_name}_basis_contract = {{}}"))
    for network_id, network in enumerate(plan.networks):
        contract_ids.setdefault(network, len(contract_ids) + 1)
    for network, contract_id in sorted(contract_ids.items(), key=lambda item: item[1]):
        lines.append(
            f"{safe_name}_basis_contract[{contract_id}] = "
            f"{_network_call(network, specs, names)}"
        )

    reconstruction = tuple(getattr(plan, "reconstruction", ()))
    lines.extend(("", "# Basis views and their exact reconstruction coefficients.",
                  f"{safe_name}_basis = {{}}"))
    coefficients = []
    for basis_index, term in enumerate(reconstruction):
        if not isinstance(term.network_id, int) or not 0 <= term.network_id < len(plan.networks):
            raise ValueError(f"reconstruction references unknown network {term.network_id!r}")
        network = plan.networks[term.network_id]
        contract_id = contract_ids[network]
        rank = len(network.output)
        permutation = tuple(term.permutation) if term.permutation else tuple(range(rank))
        if sorted(permutation) != list(range(rank)):
            raise ValueError(f"invalid reconstruction transpose {permutation!r}")
        expression = f"{safe_name}_basis_contract[{contract_id}]"
        if permutation != tuple(range(rank)):
            expression += f".transpose({', '.join(map(str, permutation))})"
        lines.append(f"# basis[{basis_index}] derives from block network {term.network_id}; transpose {permutation}.")
        lines.append(f"{safe_name}_basis[{basis_index}] = {expression}")
        coefficients.append(_number_literal(term.coefficient))
    lines.extend(("", "# Coefficients correspond one-to-one with basis views.",
                  f"{safe_name}_coefficients = [{', '.join(coefficients)}]",
                  "", "# Final block-reduced expression.",
                  f"{safe_name}_result = sum(coefficient * {safe_name}_basis[basis_index] "
                  f"for basis_index, coefficient in enumerate({safe_name}_coefficients) if coefficient)"))
    return "\n".join(lines) + "\n"


def write_block_output(plan, output_path: str | Path, *, name: str = "normal_wick_block", source=None):
    """Write :func:`render_block_output` and return basic output metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_block_output(plan, name=name, source=source), encoding="utf-8")
    return {
        "name": _py_identifier(name),
        "mode": "block",
        "output_path": str(path),
        "block_network_count": len(plan.networks),
        "w_definition_count": len(plan.w_definitions),
        "basis_view_count": len(plan.reconstruction),
        "plan": plan,
    }


__all__ = ["AdaptedNormalOutput", "MatrixRelation", "PeramMetadata",
           "load_normal_output", "project_gamma_arrays", "pairing_family_from_operators",
           "render_block_output", "write_block_output"]
