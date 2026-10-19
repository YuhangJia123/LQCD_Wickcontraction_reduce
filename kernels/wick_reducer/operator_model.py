"""Local operator and flavor model for the multi-solution reducer."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from kernels import correlator_wick_normalizer as core


ROLE_ORDER = ("bb", "b0", "0b", "00")
PERAM_ORDER = ("sin_sin", "sin_sou", "sou_sin", "sou_sou")
_DEFAULT_OPEN_SLOT = object()
_AXIS_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def make_operator(
    side: str,
    kind: str,
    species: str,
    label: str,
    flavors: Sequence[str],
    gamma: str = "",
    gamma_slots: Optional[Sequence[int]] = None,
    open_slot: object = _DEFAULT_OPEN_SLOT,
    external_tag: str = "",
    contraction_metadata: Optional[Mapping[str, Any]] = None,
    *,
    mode: str = "distillation",
    momentum: Any = None,
    projector: Any = None,
    clebsch: Any = None,
    phase: Any = None,
    external_spin: Optional[int] = None,
) -> core.Operator:
    """Build a distillation or normal operator; meson slots are (anti, q)."""
    core.validate_operator_mode(mode)
    if side not in ("sink", "source"):
        raise ValueError("side must be 'sink' or 'source'")
    if kind not in ("baryon", "meson"):
        raise ValueError("kind must be 'baryon' or 'meson'")

    flavors = tuple(flavors)
    if kind == "baryon" and len(flavors) != 3:
        raise ValueError("baryon operators require three ordered quark flavors")
    if kind == "meson":
        if len(flavors) == 1:
            flavors = (flavors[0], flavors[0])
        elif len(flavors) != 2:
            raise ValueError("meson flavors must be (antiquark_flavor, quark_flavor)")

    if open_slot is _DEFAULT_OPEN_SLOT:
        chosen_open_slot = 0 if kind == "baryon" else None
    else:
        chosen_open_slot = open_slot
    if chosen_open_slot is not None:
        if kind != "baryon" or not isinstance(chosen_open_slot, int) or not 0 <= chosen_open_slot < 3:
            raise ValueError("open_slot must be a baryon slot in {0, 1, 2}")

    arity = 3 if kind == "baryon" else 2
    if gamma_slots is None:
        if not gamma:
            slots = ()
        elif kind == "baryon":
            slots = tuple(slot for slot in range(3) if slot != chosen_open_slot)
        else:
            slots = (0, 1)
    else:
        slots = tuple(gamma_slots)
    if any(not isinstance(slot, int) or slot < 0 or slot >= arity for slot in slots):
        raise ValueError("gamma_slots contains an invalid field slot")
    if len(set(slots)) != len(slots):
        raise ValueError("gamma_slots must refer to distinct field slots")
    if gamma and len(slots) != 2:
        raise ValueError("each Gamma must connect exactly two field slots")
    if not gamma and slots:
        raise ValueError("gamma_slots require a Gamma matrix name")
    if chosen_open_slot is not None and chosen_open_slot in slots:
        raise ValueError("an open slot cannot also be a Gamma slot")

    if external_spin is not None:
        if kind != "baryon" or not isinstance(external_spin, int) or not 0 <= external_spin < 4:
            raise ValueError("external_spin must fix a baryon open slot in [0, 3]")
        if contraction_metadata is None:
            contraction_metadata = {}
        else:
            contraction_metadata = dict(contraction_metadata)
        contraction_metadata["external_spin"] = external_spin
    normalized_metadata = core.normalize_contraction_metadata(
        contraction_metadata,
        momentum=momentum,
        projector=projector,
        clebsch=clebsch,
        phase=phase,
    )
    return core.Operator(
        side, kind, species, label, flavors, gamma, slots,
        chosen_open_slot, external_tag, normalized_metadata, mode,
    )


def flavor_class(flavor: str, isospin_symmetric: bool) -> str:
    """Only u and d share a peram class in the isospin-symmetric mode."""
    normalized = flavor.lower()
    if isospin_symmetric and normalized in ("u", "d"):
        return "ud"
    return normalized


def fields_for_operators(operators: Sequence[core.Operator]):
    """Create local fields, allowing flavor-changing meson bilinears."""
    core.validate_operator_modes(operators)
    sink_axes = iter(_AXIS_NAMES)
    sink_open_count = sum(
        operator.open_slot is not None
        for operator in operators
        if operator.side == "sink"
    )
    source_open_count = sum(
        operator.open_slot is not None
        for operator in operators
        if operator.side == "source"
    )
    if sink_open_count + source_open_count > len(_AXIS_NAMES):
        raise ValueError("at most 52 bare axes are supported")
    source_axes = iter(_AXIS_NAMES[sink_open_count:])
    fields = []
    for op_index, operator in enumerate(operators):
        if operator.kind == "baryon":
            if len(operator.flavors) != 3:
                raise ValueError("baryon operators require three ordered quark flavors")
            kinds = ("q", "q", "q") if operator.side == "sink" else ("bar", "bar", "bar")
            flavors = operator.flavors
        else:
            kinds = ("bar", "q")
            if len(operator.flavors) == 1:
                flavors = (operator.flavors[0], operator.flavors[0])
            elif len(operator.flavors) == 2:
                flavors = operator.flavors
            else:
                raise ValueError("meson operator needs one legacy flavor or an (anti, quark) pair")

        for slot, (kind, flavor) in enumerate(zip(kinds, flavors)):
            bare = operator.open_slot == slot
            axis = None
            if bare:
                axis = next(sink_axes if operator.side == "sink" else source_axes)
            fields.append(core.Field(
                index=len(fields), operator=op_index, side=operator.side,
                species=operator.species, label=operator.label,
                external_tag=operator.external_tag, kind=kind, flavor=flavor,
                slot=slot, bare=bare, axis=axis,
                mode=operator.mode,
                endpoint_descriptor=core.operator_endpoint_descriptor(operator),
            ))
    quarks = [field for field in fields if field.kind == "q"]
    antiquarks = [field for field in fields if field.kind == "bar"]
    return fields, quarks, antiquarks


def profile(
    term, quarks, antiquarks, isospin_symmetric: bool,
    quark_flavor_classes: Optional[Sequence[str]] = None,
):
    """Profile perams by endpoint type, bare role, labels, and flavor class."""
    roles: Dict[str, Counter] = defaultdict(Counter)
    line_types = []
    open_routing = []
    for q_index, anti_index in enumerate(term.pairing):
        q, anti = quarks[q_index], antiquarks[anti_index]
        peram_type = core.peram_direction(q, anti)
        line_types.append(peram_type)
        flavor_tag = (
            quark_flavor_classes[q_index]
            if quark_flavor_classes is not None
            else flavor_class(q.flavor, isospin_symmetric)
        )
        peram_class = core.peram_class_label(q, anti, flavor_tag)
        sink_bare = (q.bare and q.side == "sink") or (anti.bare and anti.side == "sink")
        source_bare = (q.bare and q.side == "source") or (anti.bare and anti.side == "source")
        role = "bb" if sink_bare and source_bare else "b0" if sink_bare else "0b" if source_bare else "00"
        roles[peram_class][role] += 1
        if q.axis or anti.axis:
            open_routing.append((q.axis or "-", anti.axis or "-"))
    peram_profile = tuple(
        (name, tuple(roles[name].get(role, 0) for role in ROLE_ORDER))
        for name in sorted(roles)
    )
    type_counts = tuple((name, line_types.count(name)) for name in PERAM_ORDER)
    return peram_profile, type_counts, tuple(open_routing)


def operator_library() -> Dict[str, Tuple[core.Operator, ...]]:
    """Return built-in correlators without importing a second reducer."""

    def baryon(species: str, side: str, label: str) -> core.Operator:
        flavors = {
            "p": ("u", "u", "d"),
            "n": ("d", "d", "u"),
            "Delta++": ("u", "u", "u"),
            "Delta-": ("d", "d", "d"),
            "Omega-": ("s", "s", "s"),
        }[species]
        gamma = "Cgamma3" if species.startswith("Delta") or species == "Omega-" else "Cgamma5"
        if side == "source":
            gamma += "_dagger"
        return make_operator(
            side, "baryon", species, label, flavors, gamma, open_slot=0,
        )

    def pion(side: str, label: str) -> core.Operator:
        return make_operator(
            side, "meson", "pi0_uubar", label, ("u",), "gamma5",
        )

    ppn = (
        baryon("p", "sink", "sink_p_1"),
        baryon("p", "sink", "sink_p_2"),
        baryon("n", "sink", "sink_n"),
        make_operator(
            "source", "baryon", "p", "source_p_1", ("u", "d", "u"),
            "Cgamma5_dagger", (1, 2), open_slot=0,
        ),
        make_operator(
            "source", "baryon", "p", "source_p_2", ("u", "d", "u"),
            "Cgamma5_dagger", (1, 2), open_slot=0,
        ),
        make_operator(
            "source", "baryon", "n", "source_n", ("d", "u", "d"),
            "Cgamma5_dagger", (1, 2), open_slot=0,
        ),
    )
    omega_sink = make_operator(
        "sink", "baryon", "Omega-", "sink_Omega-", ("s", "s", "s"),
        "Cgamma3", (1, 2), open_slot=0,
    )
    omega_source = make_operator(
        "source", "baryon", "Omega-", "source_Omega-", ("s", "s", "s"),
        "Cgamma3_dagger", (1, 2), open_slot=0,
    )
    pion_plus_sink = make_operator(
        "sink", "meson", "pi+", "sink_pi+", ("d", "u"), "gamma5", (0, 1),
    )
    pion_plus_source = make_operator(
        "source", "meson", "pi+", "source_pi+", ("u", "d"), "gamma5", (0, 1),
    )
    return {
        "pipi": (
            pion("sink", "sink_pi_1"), pion("sink", "sink_pi_2"),
            pion("source", "source_pi_1"), pion("source", "source_pi_2"),
        ),
        "delta": (
            baryon("Delta++", "sink", "sink_Delta++"),
            baryon("Delta++", "source", "source_Delta++"),
        ),
        "pn": (
            baryon("p", "sink", "sink_p"), baryon("n", "sink", "sink_n"),
            baryon("p", "source", "source_p"), baryon("n", "source", "source_n"),
        ),
        "ppn": ppn,
        "pip": (
            pion("sink", "sink_pi"), baryon("p", "sink", "sink_p"),
            pion("source", "source_pi"), baryon("p", "source", "source_p"),
        ),
        "pip-p": (
            baryon("p", "sink", "sink_p"), pion("source", "source_pi"),
            baryon("p", "source", "source_p"),
        ),
        "delta++delta-": (
            baryon("Delta++", "sink", "sink_Delta++"),
            baryon("Delta-", "sink", "sink_Delta-"),
            baryon("Delta++", "source", "source_Delta++"),
            baryon("Delta-", "source", "source_Delta-"),
        ),
        "pn_omega_pi+": (
            baryon("p", "sink", "sink_p"),
            baryon("n", "sink", "sink_n"),
            omega_sink,
            pion_plus_sink,
            baryon("p", "source", "source_p"),
            baryon("n", "source", "source_n"),
            omega_source,
            pion_plus_source,
        ),
    }
