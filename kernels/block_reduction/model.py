"""Project-independent symbolic inputs and reusable block plans.

Indices are opaque strings, not operator or Wick identifiers.  The only
special tensor is a full-spin normal propagator with two color/spin ports in
the unified ``(V_sink,D_sink,V_source,D_source)`` layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import isfinite
from operator import index as integer_index
import sys
from typing import Any, Hashable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True, eq=False)
class GaussianRational:
    """An exact complex coefficient, including binary floating-point inputs."""
    real: Fraction
    imag: Fraction

    def __post_init__(self):
        try:
            object.__setattr__(self, "real", Fraction(self.real))
            object.__setattr__(self, "imag", Fraction(self.imag))
        except (ValueError, OverflowError, TypeError) as error:
            raise ValueError("complex coefficient components must be finite rational scalars") from error

    def __complex__(self):
        return complex(float(self.real), float(self.imag))

    def __bool__(self):
        return bool(self.real or self.imag)

    def __neg__(self):
        return GaussianRational(-self.real, -self.imag)

    def __add__(self, other):
        other = exact_scalar(other)
        if isinstance(other, GaussianRational):
            return GaussianRational(self.real + other.real, self.imag + other.imag)
        return GaussianRational(self.real + other, self.imag)

    __radd__ = __add__

    def __sub__(self, other):
        return self + -exact_scalar(other)

    def __rsub__(self, other):
        return exact_scalar(other) + -self

    def __mul__(self, other):
        other = exact_scalar(other)
        if isinstance(other, GaussianRational):
            return GaussianRational(self.real * other.real - self.imag * other.imag,
                                    self.real * other.imag + self.imag * other.real)
        return GaussianRational(self.real * other, self.imag * other)

    __rmul__ = __mul__

    def __eq__(self, other):
        # Fraction accepts numeric strings as constructor input, but strings
        # are not numerically equal to rational or complex scalar objects.
        if isinstance(other, (str, bytes, bytearray)):
            return False
        try:
            other = exact_scalar(other)
        except (ValueError, TypeError):
            return False
        if isinstance(other, GaussianRational):
            return self.real == other.real and self.imag == other.imag
        return self.imag == 0 and self.real == other

    def __hash__(self):
        # Match Python's numeric hash rule.  These objects compare equal to
        # built-in complex values when both components are exactly equal, so
        # hashing a pair of Fractions would violate equality's hash contract.
        value = hash(self.real) + sys.hash_info.imag * hash(self.imag)
        modulus = 1 << sys.hash_info.width
        value = (value + modulus // 2) % modulus - modulus // 2
        return -2 if value == -1 else value


def exact_scalar(value):
    """Do not discard small terms or introduce rounding in coefficient ledgers."""
    if isinstance(value, (Fraction, GaussianRational)):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, complex):
        if not isfinite(value.real) or not isfinite(value.imag):
            raise ValueError("coefficients must be finite")
        real, imag = Fraction(value.real), Fraction(value.imag)
        return GaussianRational(real, imag) if imag else real
    try:
        return Fraction(value)
    except (ValueError, OverflowError, TypeError) as error:
        raise ValueError("coefficients must be finite rational or numeric scalars") from error


@dataclass(frozen=True)
class TensorSymmetry:
    permutation: Tuple[int, ...]
    sign: int = 1

    def __post_init__(self):
        try:
            object.__setattr__(self, "permutation", tuple(integer_index(axis) for axis in self.permutation))
        except TypeError as error:
            raise ValueError("a tensor symmetry must contain integer axis positions") from error
        if sorted(self.permutation) != list(range(len(self.permutation))):
            raise ValueError("a tensor symmetry must be a permutation")
        if self.sign not in (-1, 1):
            raise ValueError("tensor symmetry signs must be +1 or -1")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    axis_types: Tuple[str, ...]
    kind: str = "tensor"
    identity: Optional[str] = None
    mode: str = "normal"
    symmetries: Tuple[TensorSymmetry, ...] = ()
    peram_ports: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

    def __post_init__(self):
        try:
            object.__setattr__(self, "shape", tuple(integer_index(size) for size in self.shape))
        except TypeError as error:
            raise ValueError("tensor dimensions must be positive integers") from error
        object.__setattr__(self, "axis_types", tuple(self.axis_types))
        object.__setattr__(self, "symmetries", tuple(
            s if isinstance(s, TensorSymmetry) else TensorSymmetry(**s)
            for s in self.symmetries
        ))
        if len(self.shape) != len(self.axis_types) or any(d <= 0 for d in self.shape):
            raise ValueError("tensor shape and axis_types must have matching positive dimensions")
        if self.kind not in ("tensor", "peram", "wedge"):
            raise ValueError("tensor kind must be tensor, peram, or wedge")
        for symmetry in self.symmetries:
            p = symmetry.permutation
            if len(p) != len(self.shape) or any(
                (self.shape[i], self.axis_types[i]) != (self.shape[j], self.axis_types[j])
                for i, j in enumerate(p)
            ):
                raise ValueError("tensor symmetry must preserve axis types and dimensions")
        if self.peram_ports is not None:
            try:
                object.__setattr__(self, "peram_ports", tuple(tuple(integer_index(axis) for axis in p) for p in self.peram_ports))
            except TypeError as error:
                raise ValueError("peram_ports must contain integer axis positions") from error
        elif self.kind == "peram":
            # Every peram uses the project-wide (V_sink,D_sink,V_source,D_source)
            # layout. The vector/eigen axis is represented by the project
            # spatial axis type, and each endpoint port is ordered (V, D).
            color = [i for i, t in enumerate(self.axis_types) if t == "color"]
            spin = [i for i, t in enumerate(self.axis_types) if t == "spin"]
            if len(spin) == len(color) == 2:
                object.__setattr__(self, "peram_ports", ((color[0], spin[0]), (color[1], spin[1])))
        if self.kind == "peram":
            ports = self.peram_ports
            if ports is None or len(ports) != 2 or any(len(p) != 2 for p in ports):
                raise ValueError("a peram needs two endpoint ports")
            if sorted(i for port in ports for i in port) != list(range(4)):
                raise ValueError("peram ports must partition its four axes")
            if any(self.axis_types[v] != "color" or self.axis_types[d] != "spin" for v, d in ports):
                raise ValueError("peram_ports must list V/color then D/spin for each port")


@dataclass(frozen=True)
class Factor:
    tensor: str
    indices: Tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "indices", tuple(self.indices))


@dataclass(frozen=True)
class Network:
    factors: Tuple[Factor, ...]
    output: Tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "output", tuple(self.output))

    @classmethod
    def from_einsum(cls, equation, operands):
        equation = "".join(equation.split())
        if equation.count("->") != 1 or "." in equation:
            raise ValueError("symbolic einsum requires explicit output and no ellipsis")
        inputs, output = equation.split("->")
        slots = inputs.split(",")
        if len(slots) != len(operands):
            raise ValueError("einsum operands do not match its input slots")
        return cls(tuple(Factor(name, tuple(axes)) for name, axes in zip(operands, slots)), tuple(output))


@dataclass(frozen=True)
class BasisView:
    contract_id: Hashable
    scalar: Any = 1
    permutation: Tuple[int, ...] = ()
    coefficient: Any = 1

    def __post_init__(self):
        object.__setattr__(self, "permutation", tuple(self.permutation))


@dataclass(frozen=True)
class WeightedNetwork:
    network: Network
    coefficient: Any


@dataclass(frozen=True)
class BlockDefinition:
    name: str
    propagator: str
    # Each joint endpoint is ordered (V, D) for normal perams.
    ports: Tuple[Tuple[int, int], Tuple[int, int]]
    shape: Tuple[int, ...] = (3, 4, 3, 4, 3, 4, 3, 4)


@dataclass(frozen=True)
class ReconstructionTerm:
    network_id: int
    coefficient: Any
    permutation: Tuple[int, ...]


@dataclass(frozen=True)
class RewriteStep:
    """Exact local identity; factor indices refer to the stored before network."""
    before: Tuple[WeightedNetwork, ...]
    after: Tuple[WeightedNetwork, ...]
    block: BlockDefinition
    factor_positions: Tuple[int, int]


@dataclass(frozen=True)
class FamilyCertificate:
    """A supplied pairing sum and one exact simultaneous W rewrite."""
    raw_terms: Sequence[WeightedNetwork]
    fused_terms: Tuple[WeightedNetwork, ...]
    pairs: Tuple[Tuple[int, int], ...]
    factor_positions: Tuple[int, ...]
    w_definitions: Tuple[BlockDefinition, ...] = ()


@dataclass(frozen=True)
class BlockPlan:
    w_definitions: Tuple[BlockDefinition, ...]
    networks: Tuple[Network, ...]
    reconstruction: Tuple[ReconstructionTerm, ...]
    tensor_specs: Mapping[str, TensorSpec]
    original_terms: Tuple[WeightedNetwork, ...]
    rewrites: Tuple[RewriteStep, ...]
    stats: Mapping[str, Any] = field(default_factory=dict)
    family_certificate: Optional[FamilyCertificate] = None

    def verify(self):
        from .planner import verify_plan
        return verify_plan(self)

    @property
    def certificate(self):
        """The plan is its own independently replayable symbolic certificate."""
        return self


def as_network(value):
    if isinstance(value, Network):
        return value
    if isinstance(value, Mapping):
        return Network.from_einsum(value["equation"], value["operands"])
    if isinstance(value, tuple) and len(value) == 2:
        return Network.from_einsum(*value)
    raise TypeError("basis_contract entries must be symbolic Network or equation/operands definitions")


def as_view(value):
    if isinstance(value, BasisView):
        return value
    if isinstance(value, Mapping):
        return BasisView(**value)
    raise TypeError("basis entries must be BasisView or its field mapping")


def compose_transposes(p, q):
    """T.transpose(p).transpose(q) == T.transpose(compose_transposes(p, q))."""
    if sorted(p) != list(range(len(p))) or sorted(q) != list(range(len(p))):
        raise ValueError("invalid transpose permutations")
    return tuple(p[i] for i in q)
