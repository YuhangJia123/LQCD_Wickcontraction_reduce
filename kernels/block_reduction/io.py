"""Read generated tensor expressions as data, without executing Python source."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .model import BasisView, Factor, Network, TensorSpec


@dataclass(frozen=True)
class SymbolicInput:
    """The three expression tables, with explicit caller-supplied tensor types.

    ``basis`` stores scalar factors and composed array transpositions.
    ``coefficients`` remains a separate target vector, as in the source file.
    No operator modes or tensor symmetries are inferred from that file.
    """

    basis_contract: Mapping[int, Network]
    basis: Mapping[int, BasisView]
    coefficients: tuple[complex, ...]
    tensor_specs: Mapping[str, TensorSpec]
    prefix: str = ""


def _error(node: ast.AST, message: str) -> ValueError:
    line = getattr(node, "lineno", "?")
    return ValueError(f"line {line}: {message}")


def _scalar(node: ast.AST) -> int | float | complex:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, complex):
        result = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        result = _scalar(node.operand)
        if isinstance(node.op, ast.USub):
            result = -result
    elif isinstance(node, ast.BinOp):
        left, right = _scalar(node.left), _scalar(node.right)
        try:
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            else:
                raise _error(node, "unsupported scalar arithmetic; use +, -, *, or /")
        except (OverflowError, ZeroDivisionError) as exc:
            raise _error(node, f"invalid scalar arithmetic: {exc}") from exc
    else:
        raise _error(node, "expected a numeric literal or scalar arithmetic")
    try:
        finite = math.isfinite(result.real) and math.isfinite(result.imag)
    except OverflowError:
        finite = isinstance(result, int)
    if not finite:
        raise _error(node, "scalar values must be finite")
    return result


def _integer(node: ast.AST, description: str) -> int:
    value = _scalar(node)
    if value.imag or value.real != int(value.real):
        raise _error(node, f"{description} must be an integer literal")
    return int(value.real)


def _table_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        return target.value.id
    return None


def _table_root(node: ast.AST) -> str | None:
    """Follow a table entry or attribute back to its named table."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _check_table_access(statement: ast.AST, names: Mapping[str, str]) -> None:
    """Reject hidden table changes, including assignment right-hand sides.

    The three accepted tables are a static interchange format, not a Python
    program.  Letting an unrelated assignment hide append/update or a walrus
    rebind would otherwise return a different target from the source tables.
    """
    allowed_target = None
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        if _table_name(target) in names:
            allowed_target = target
    readonly_methods = {"get", "items", "keys", "values", "copy", "__getitem__", "transpose"}
    readonly_functions = {"sum", "enumerate", "len", "zip", "tuple", "list", "dict", "sorted", "reversed"}
    parents = {id(child): node for node in ast.walk(statement) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(statement):
        if isinstance(node, (ast.Name, ast.Subscript, ast.Attribute)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if _table_root(node) in names and node is not allowed_target:
                raise _error(node, "expression table writes must be plain top-level assignments")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            raise _error(node, "rebinding an expression table is unsupported")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if bound in names:
                    raise _error(node, "rebinding an expression table through import is unsupported")
        if isinstance(node, ast.ExceptHandler) and node.name in names:
            raise _error(node, "rebinding an expression table is unsupported")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names:
            parent = parents.get(id(node))
            is_entry = isinstance(parent, ast.Subscript) and parent.value is node
            is_method = isinstance(parent, ast.Attribute) and parent.value is node
            is_consumer = isinstance(parent, ast.Call) and node in parent.args
            if not (is_entry or is_method or is_consumer):
                raise _error(node, "escaping or aliasing expression tables is unsupported")
        if isinstance(node, ast.Attribute) and _table_root(node.value) in names:
            parent = parents.get(id(node))
            if not (isinstance(parent, ast.Call) and parent.func is node):
                raise _error(node, "escaping expression table methods is unsupported")
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and _table_root(node.func.value) in names:
            if node.func.attr not in readonly_methods:
                raise _error(node, "mutating or unsupported expression table methods are unsupported")
        # Unknown functions can mutate a table passed as an argument.  Only
        # established read-only consumers used by generated result expressions
        # are allowed to receive one of the table objects directly.
        passes_table = any(_table_root(arg) in names for arg in node.args)
        passes_table |= any(_table_root(keyword.value) in names for keyword in node.keywords)
        if passes_table and not (isinstance(node.func, ast.Name) and node.func.id in readonly_functions):
            raise _error(node, "passing an expression table to an unsupported function is unsupported")
    if isinstance(statement, ast.Assign) and _table_root(statement.value) in names:
        # A mutable alias followed by an indirect write cannot be interpreted
        # as a static table.  Copy methods are separate read-only operations.
        if statement.targets[0] is not allowed_target:
            raise _error(statement, "aliasing expression tables is unsupported")


def _parse_contract(node: ast.AST, specs: Mapping[str, TensorSpec]) -> Network:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "contract"):
        raise _error(node, "basis_contract entries must be direct contract(...) calls")
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        raise _error(node, "contract requires a literal, explicit einsum equation")
    equation = "".join(node.args[0].value.split())
    if equation.count("->") != 1 or "..." in equation:
        raise _error(node, "einsum must have one explicit output and no ellipsis")
    left, output = equation.split("->")
    inputs = left.split(",")
    if len(inputs) != len(node.args) - 1:
        raise _error(node, "einsum operand count does not match the equation")
    if any(not part.isascii() or any(not char.isalpha() for char in part) for part in (*inputs, output)):
        raise _error(node, "einsum indices must be ASCII letters")
    for keyword in node.keywords:
        if keyword.arg != "optimize":
            raise _error(keyword, "only the optimize keyword is supported on contract")
        # A path selection does not change the symbolic network.  Check that
        # its syntax is literal rather than executing or interpreting it.
        try:
            ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise _error(keyword, "optimize must be a literal") from exc
    factors = []
    dimensions: dict[str, int] = {}
    types: dict[str, str] = {}
    for operand, indices in zip(node.args[1:], inputs):
        if not isinstance(operand, ast.Name):
            raise _error(operand, "tensor operands must be names; array calls, transposes, and starred operands are unsupported")
        if operand.id not in specs:
            raise _error(operand, f"missing explicit TensorSpec for {operand.id!r}")
        spec = specs[operand.id]
        if len(indices) != len(spec.shape) or len(spec.axis_types) != len(spec.shape):
            raise _error(operand, f"tensor rank for {operand.id!r} disagrees with the equation")
        for label, size, axis_type in zip(indices, spec.shape, spec.axis_types):
            if label in dimensions and dimensions[label] != size:
                raise _error(operand, f"index {label!r} has incompatible dimensions")
            if label in types and types[label] != axis_type:
                raise _error(operand, f"index {label!r} has incompatible axis types")
            dimensions[label], types[label] = size, axis_type
        factors.append(Factor(operand.id, tuple(indices)))
    if len(set(output)) != len(output) or any(label not in dimensions for label in output):
        raise _error(node, "output indices must be distinct existing tensor indices")
    return Network(tuple(factors), tuple(output))


def _parse_view(node: ast.AST, name: str, contracts: Mapping[int, Network]) -> BasisView:
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == name:
        contract_id = _integer(node.slice, "contract ID")
        if contract_id not in contracts:
            raise _error(node, f"basis references undefined contract {contract_id}")
        return BasisView(contract_id, permutation=tuple(range(len(contracts[contract_id].output))))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        view = _parse_view(node.operand, name, contracts)
        sign = -1 if isinstance(node.op, ast.USub) else 1
        return BasisView(view.contract_id, sign * view.scalar, view.permutation)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        if isinstance(node.op, ast.Div):
            view, scalar = _parse_view(node.left, name, contracts), _scalar(node.right)
            if not scalar:
                raise _error(node, "a basis scalar divisor cannot be zero")
            scalar = 1 / scalar
        else:
            try:
                scalar = _scalar(node.left)
            except ValueError:
                view, scalar = _parse_view(node.left, name, contracts), _scalar(node.right)
            else:
                view = _parse_view(node.right, name, contracts)
        return BasisView(view.contract_id, scalar * view.scalar, view.permutation)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "transpose":
        if node.keywords:
            raise _error(node, "transpose keyword arguments are unsupported")
        view = _parse_view(node.func.value, name, contracts)
        rank = len(view.permutation)
        args = node.args
        if not args:
            permutation = tuple(reversed(range(rank)))
        elif len(args) == 1 and isinstance(args[0], (ast.Tuple, ast.List)):
            permutation = tuple(_integer(value, "transpose axis") for value in args[0].elts)
        else:
            permutation = tuple(_integer(value, "transpose axis") for value in args)
        permutation = tuple(axis + rank if axis < 0 else axis for axis in permutation)
        if sorted(permutation) != list(range(rank)):
            raise _error(node, "transpose must contain each output axis exactly once")
        composed = tuple(view.permutation[axis] for axis in permutation)
        return BasisView(view.contract_id, view.scalar, composed)
    raise _error(node, "unsupported basis expression; expected a contract reference, scalar factor, or transpose")


def load_symbolic_output(source_or_path: str | Path, *, tensor_specs: Mapping[str, TensorSpec], prefix: str | None = None) -> SymbolicInput:
    """Parse generated ``*_basis_contract``, ``*_basis``, and coefficient tables.

    Only top-level assignments to those three tables are interpreted.  Imports,
    result expressions, and unrelated code are never executed.  Tensor shapes,
    normal/distillation mode, identities, and symmetries come exclusively from
    ``tensor_specs``; a file containing VVV factors is not silently converted
    into a normal-mode color contraction.
    """
    if isinstance(source_or_path, Path):
        source = source_or_path.read_text(encoding="utf-8")
    elif isinstance(source_or_path, str):
        source = source_or_path
        if "\n" not in source and "\r" not in source:
            try:
                path = Path(source)
                if path.is_file():
                    source = path.read_text(encoding="utf-8")
            except OSError:
                pass
    else:
        raise TypeError("source_or_path must be Python source text or a pathlib.Path")
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"invalid symbolic output source: {exc}") from exc
    specs = dict(tensor_specs)
    candidates = set()
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                name = _table_name(target)
                if name and name.endswith("_basis_contract"):
                    candidates.add(name[:-len("_basis_contract")])
    if prefix is None:
        if len(candidates) != 1:
            raise ValueError(f"expected one expression prefix, found {sorted(candidates)!r}; pass prefix explicitly")
        prefix = next(iter(candidates))
    names = {f"{prefix}_basis_contract": "contracts", f"{prefix}_basis": "basis", f"{prefix}_coefficients": "coefficients"}
    entries: dict[str, dict[int, ast.AST]] = {"contracts": {}, "basis": {}}
    initialized: set[str] = set()
    coefficients: tuple[complex, ...] | None = None
    for statement in module.body:
        _check_table_access(statement, names)
        if not isinstance(statement, ast.Assign):
            continue
        relevant = [target for target in statement.targets if _table_name(target) in names]
        if not relevant:
            continue
        if len(statement.targets) != 1:
            raise _error(statement, "multiple-target expression table assignments are unsupported")
        target = relevant[0]
        kind = names[_table_name(target)]
        if kind == "coefficients":
            if not isinstance(target, ast.Name) or not isinstance(statement.value, (ast.List, ast.Tuple)):
                raise _error(statement, "coefficients must be assigned a literal list or tuple")
            if coefficients is not None:
                raise _error(statement, "duplicate coefficients assignment")
            coefficients = tuple(_scalar(value) for value in statement.value.elts)
        elif isinstance(target, ast.Name):
            if kind in initialized or entries[kind]:
                raise _error(statement, f"duplicate {kind} table initialization")
            if not isinstance(statement.value, ast.Dict):
                raise _error(statement, f"{kind} must be initialized with a dictionary")
            initialized.add(kind)
            for key, value in zip(statement.value.keys, statement.value.values):
                if key is None:
                    raise _error(statement, "dictionary unpacking is unsupported")
                index = _integer(key, "table index")
                if index in entries[kind]:
                    raise _error(key, "duplicate table index")
                entries[kind][index] = value
        else:
            if kind not in initialized:
                raise _error(statement, f"{kind} table must be initialized before indexed assignments")
            index = _integer(target.slice, "table index")
            if index in entries[kind]:
                raise _error(statement, "duplicate table index")
            entries[kind][index] = statement.value
    if coefficients is None or initialized != {"contracts", "basis"}:
        raise ValueError(f"missing required expression tables for prefix {prefix!r}")
    contracts = {index: _parse_contract(value, specs) for index, value in entries["contracts"].items()}
    basis = {index: _parse_view(value, f"{prefix}_basis_contract", contracts) for index, value in entries["basis"].items()}
    if sorted(basis) != list(range(len(coefficients))):
        raise ValueError("basis keys must be contiguous from zero and match the coefficients length")
    return SymbolicInput(MappingProxyType(contracts), MappingProxyType(basis), coefficients, MappingProxyType(specs), prefix)


__all__ = ["SymbolicInput", "load_symbolic_output"]
