"""Planner for the normal-wick-block raw pairing reduction.

The search space consists of exact antisymmetric two-propagator rewrites.
Small expressions use individual branches.  Large expressions use batches
of compatible rewrites, with explicit reporting of the restricted search.
Neither a search heuristic nor a numerical sample is an equivalence proof:
the returned plan is checked by expanding its fixed W definitions.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
import heapq
from itertools import combinations
from math import isfinite
from numbers import Real
from time import monotonic
from typing import Mapping

from .canonical import (OrbitCanonicalizer, canonicalization_cache,
                        canonicalize as _graph_canonicalize)
from .model import (
    BasisView, BlockDefinition, BlockPlan, FamilyCertificate, Factor, Network, ReconstructionTerm,
    RewriteStep, TensorSpec, TensorSymmetry, WeightedNetwork, as_network,
    as_view, exact_scalar,
)


_canonicalizers = ContextVar("block_reduction_orbit_canonicalizers", default=None)


class _BoundedCache(OrderedDict):
    """Per-plan memoization; eviction changes runtime but never equivalence."""

    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def get(self, key, default=None):
        value = super().get(key, default)
        if key in self:
            self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.capacity:
            self.popitem(last=False)


def canonicalize(network, specs, *, fixed_output=True):
    engines = _canonicalizers.get()
    if engines is None:
        return _graph_canonicalize(network, specs, fixed_output=fixed_output)
    for engine in engines:
        if engine.specs == specs:
            return engine.canonicalize(network, fixed_output=fixed_output)
    engine = OrbitCanonicalizer(specs)
    engines.append(engine)
    return engine.canonicalize(network, fixed_output=fixed_output)


def normalize_inputs(basis_contract, basis, coefficients, tensor_specs):
    """Read the final expression; no original Wick or reducer metadata is needed."""
    specs = {
        name: spec if isinstance(spec, TensorSpec) else TensorSpec(name=name, **spec)
        for name, spec in tensor_specs.items()
    }
    if any(spec.name != name for name, spec in specs.items()):
        raise ValueError("TensorSpec names must match tensor_specs keys")
    contracts = {key: as_network(value) for key, value in basis_contract.items()}
    source_views = basis.items() if isinstance(basis, Mapping) else enumerate(basis)
    views = {key: as_view(value) for key, value in source_views}
    terms = []
    output_space = None
    for key, view in views.items():
        if view.contract_id not in contracts:
            raise ValueError(f"basis view {key!r} references a missing contraction")
        network = contracts[view.contract_id]
        p = view.permutation or tuple(range(len(network.output)))
        if sorted(p) != list(range(len(network.output))):
            raise ValueError(f"basis view {key!r} has an invalid transpose")
        external = 1 if coefficients is None else coefficients[key]
        coefficient = exact_scalar(external) * exact_scalar(view.coefficient) * exact_scalar(view.scalar)
        viewed = Network(network.factors, tuple(network.output[i] for i in p))
        space = _output_space(viewed, specs)
        if output_space is not None and space != output_space:
            raise ValueError("all basis views must produce the same ordered output spaces")
        output_space = space
        views[key] = BasisView(view.contract_id, view.scalar, p, view.coefficient)
        if coefficient:
            terms.append(WeightedNetwork(viewed, coefficient))
    # Validate even unused definitions: mistyped slots must not be hidden by a
    # zero coefficient.  Canonicalization performs the full space validation.
    for network in contracts.values():
        _output_space(network, specs)
    return contracts, views, tuple(terms), specs


def _output_space(network, specs):
    spaces = {}
    for factor in network.factors:
        if factor.tensor not in specs:
            raise ValueError(f"missing TensorSpec for operand {factor.tensor!r}")
        spec = specs[factor.tensor]
        if len(factor.indices) != len(spec.shape):
            raise ValueError("operand index count does not match TensorSpec shape")
        for label, shape, axis_type in zip(factor.indices, spec.shape, spec.axis_types):
            space = (axis_type, shape)
            if label in spaces and spaces[label] != space:
                raise ValueError(f"index {label!r} connects incompatible tensor spaces")
            spaces[label] = space
    if len(set(network.output)) != len(network.output) or any(i not in spaces for i in network.output):
        raise ValueError("output indices must be distinct and occur in the network")
    return tuple(spaces[i] for i in network.output)


def _normal_only(specs):
    for spec in specs.values():
        if spec.mode != "normal":
            raise ValueError("block_reduce supports only normal tensors, not distillation")
        if any(t == "spin" and d != 4 for t, d in zip(spec.axis_types, spec.shape)):
            raise ValueError("block_reduce requires all four Dirac components; upper2 is forbidden")
        if spec.kind == "wedge":
            raise ValueError("input W tensors are not supported; supply ordinary basis contractions")
        if spec.kind == "peram":
            if len(spec.shape) != 4 or any(
                spec.shape[v] != 3 or spec.shape[d] != 4
                or spec.axis_types[v] != "color" or spec.axis_types[d] != "spin"
                for v, d in spec.peram_ports
            ):
                raise ValueError(
                    "normal propagator ports require (color, spin)=(3, 4) at both ends"
                )
    identities = {}
    for spec in specs.values():
        identity = spec.name if spec.identity is None else spec.identity
        shape = (spec.shape, spec.axis_types, spec.kind, spec.peram_ports, spec.symmetries)
        if identity in identities and identities[identity] != shape:
            raise ValueError("tensor aliases must have matching slot layouts and symmetries")
        identities[identity] = shape


def _block_specs(specs):
    """Prepare all possible types once, keeping canonical cache keys stable."""
    expanded = dict(specs)
    aliases = {}
    for name in sorted(specs):
        spec = specs[name]
        if spec.kind == "peram":
            aliases.setdefault(spec.name if spec.identity is None else spec.identity, name)
    definitions = {}
    for propagator in aliases.values():
        spec = specs[propagator]
        base = "W_" + propagator
        name = base
        suffix = 0
        while name in expanded:
            suffix += 1
            name = base + "_" + str(suffix)
        definition = BlockDefinition(name, propagator, spec.peram_ports)
        definitions[name] = definition
        expanded[name] = TensorSpec(
            name, definition.shape, ("color", "spin") * 4, kind="wedge",
            symmetries=(
                TensorSymmetry((2, 3, 0, 1, 4, 5, 6, 7), -1),
                TensorSymmetry((0, 1, 2, 3, 6, 7, 4, 5), -1),
            ),
        )
    by_propagator = {}
    for name, spec in specs.items():
        if spec.kind == "peram":
            by_propagator[name] = next(
                d for d in definitions.values()
                if (specs[d.propagator].name if specs[d.propagator].identity is None
                    else specs[d.propagator].identity) ==
                   (spec.name if spec.identity is None else spec.identity)
            )
    return expanded, definitions, by_propagator


def _collect(terms, specs):
    ledger = {}
    for term in terms:
        if not term.coefficient:
            continue
        canonical = canonicalize(term.network, specs, fixed_output=True)
        coefficient = exact_scalar(term.coefficient) * canonical.scalar
        if not coefficient:
            continue
        old = ledger.get(canonical.key)
        coefficient += old.coefficient if old else 0
        if coefficient:
            ledger[canonical.key] = WeightedNetwork(canonical.network, coefficient)
        else:
            ledger.pop(canonical.key, None)
    return ledger


def _collect_family(family, specs):
    """Use proven endpoint actions only for the exact compact family law."""
    from .global_layout import CompactPairingTerms, _collect_compact_family
    if type(family.terms) is CompactPairingTerms:
        engines = _canonicalizers.get()
        engine = None if engines is None else next(
            (candidate for candidate in engines if candidate.specs == specs), None)
        if engine is None:
            engine = OrbitCanonicalizer(specs)
            if engines is not None:
                engines.append(engine)
        ordinary_specs = {name: spec for name, spec in specs.items()
                          if spec.kind != "wedge"}
        return _collect_compact_family(family, ordinary_specs, canonicalizer=engine)
    return _collect(family.terms, specs)


def _used_types(ledger, definitions):
    return frozenset(
        factor.tensor for term in ledger.values() for factor in term.network.factors
        if factor.tensor in definitions
    )


def _state_key(ledger):
    return frozenset((key, term.coefficient) for key, term in ledger.items())


def _reconstruct(ledger, specs, free_cache):
    networks = {}
    weights = defaultdict(lambda: 0)
    for term in ledger.values():
        canonical = free_cache.get(term.network)
        if canonical is None:
            canonical = canonicalize(term.network, specs, fixed_output=False)
            free_cache[term.network] = canonical
        coefficient = term.coefficient * canonical.scalar
        if coefficient:
            networks.setdefault(canonical.key, canonical.network)
            weights[canonical.key, canonical.permutation] += coefficient
    live = {key for (key, _p), c in weights.items() if c}
    ordered = sorted(live, key=repr)
    positions = {key: i for i, key in enumerate(ordered)}
    reconstruction = tuple(
        ReconstructionTerm(positions[key], coefficient, p)
        for (key, p), coefficient in sorted(weights.items(), key=lambda item: repr(item[0]))
        if coefficient
    )
    return tuple(networks[key] for key in ordered), reconstruction


def _score(ledger, specs, free_cache):
    # Search needs a count, not sorted network IDs and ReconstructionTerm
    # objects. Preserve the same signed view ledger, without materializing
    # the output plan at every branch.
    weights = defaultdict(lambda: 0)
    primitive_counts = {}
    for term in ledger.values():
        canonical = free_cache.get(term.network)
        if canonical is None:
            canonical = canonicalize(term.network, specs, fixed_output=False)
            free_cache[term.network] = canonical
        coefficient = term.coefficient * canonical.scalar
        if coefficient:
            weights[canonical.key, canonical.permutation] += coefficient
            if canonical.key not in primitive_counts:
                primitive_counts[canonical.key] = sum(
                    specs[f.tensor].kind == "peram" for f in canonical.network.factors
                )
    live = {key for (key, _permutation), value in weights.items() if value}
    # The first coordinate is the objective.  Fewer unfused factors is only
    # a deterministic tie-break, allowing useful equal-count W plans.
    return len(live), sum(primitive_counts[key] for key in live), len(ledger)


@dataclass(frozen=True)
class _Candidate:
    x_key: object
    y_key: object
    z_key: object
    z: Network
    z_sign: int
    x_view: Network
    y_view: Network
    x_sign: int
    y_sign: int
    definition: BlockDefinition
    positions: tuple
    family: object


def _cut(network, i, j, spec, definition):
    first, second = network.factors[i], network.factors[j]
    left, right = spec.peram_ports
    a = tuple(first.indices[k] for k in left)
    b = tuple(second.indices[k] for k in left)
    source_i = tuple(first.indices[k] for k in right)
    source_j = tuple(second.indices[k] for k in right)
    indices_i, indices_j = list(first.indices), list(second.indices)
    for k, li, lj in zip(right, source_i, source_j):
        indices_i[k], indices_j[k] = lj, li
    exchange = list(network.factors)
    exchange[i] = Factor(first.tensor, tuple(indices_i))
    exchange[j] = Factor(second.tensor, tuple(indices_j))
    residual = tuple(f for k, f in enumerate(network.factors) if k not in (i, j))
    fused = Network(residual + (Factor(definition.name, a + b + source_i + source_j),), network.output)
    return Network(tuple(exchange), network.output), fused


def _candidates(ledger, specs, definitions, by_propagator, max_types,
                free_cache, deadline, allow_bridge, cut_cache=None):
    """Index exchange partners while preserving their residual by construction.

    Cuts are generated once per compute-network orbit, not once per basis
    transpose.  The swapped-network hash locates a partner with the same
    residual and four-port boundary; fixed-output forms check its actual
    target axes.  Equivalent cuts are deduplicated by their exact result.
    """
    groups = defaultdict(list)
    for key, term in ledger.items():
        c = free_cache.get(term.network)
        if c is None:
            c = canonicalize(term.network, specs, fixed_output=False)
            free_cache[term.network] = c
        if c.scalar:
            groups[c.key].append((key, term, c))
    used = _used_types(ledger, definitions)
    candidates = {}
    interrupted = False
    for group_key in sorted(groups, key=repr):
        if monotonic() >= deadline:
            interrupted = True
            break
        members = groups[group_key]
        base = members[0][2].network
        propagators = [i for i, f in enumerate(base.factors) if f.tensor in by_propagator]
        if max_types == 1:
            # A single W definition can pair only factors in its numerical
            # identity class. Once selected, skip every other class entirely.
            chosen = next(iter(used), None)
            by_type = defaultdict(list)
            for i in propagators:
                definition = by_propagator[base.factors[i].tensor]
                if chosen is None or definition.name == chosen:
                    by_type[definition.name].append(i)
            pairs = (pair for positions in by_type.values()
                     for pair in combinations(positions, 2))
        else:
            pairs = combinations(propagators, 2)
        for i, j in pairs:
            if monotonic() >= deadline:
                interrupted = True
                break
            first, second = base.factors[i], base.factors[j]
            definition = by_propagator[first.tensor]
            if by_propagator[second.tensor] != definition:
                continue
            if definition.name not in used and len(used) >= max_types:
                continue
            cache_key = (base, i, j)
            cached = None if cut_cache is None else cut_cache.get(cache_key)
            if cached is None:
                exchange, fused = _cut(base, i, j, specs[first.tensor], definition)
                cy_free = canonicalize(exchange, specs, fixed_output=False)
                cz_free = None
                if cut_cache is not None:
                    cut_cache[cache_key] = (exchange, fused, cy_free, cz_free)
            else:
                exchange, fused, cy_free, cz_free = cached
            if not cy_free.scalar:
                continue
            has_partner_orbit = cy_free.key in groups
            if not has_partner_orbit and not allow_bridge:
                continue
            if cz_free is None:
                cz_free = canonicalize(fused, specs, fixed_output=False)
                if cut_cache is not None:
                    cut_cache[cache_key] = (exchange, fused, cy_free, cz_free)
            if not cz_free.scalar:
                continue
            family = (frozenset((group_key, cy_free.key)), cz_free.key, definition.name)
            for x_key, _term, cx in members:
                output = tuple(base.output[k] for k in cx.permutation)
                x_view = Network(base.factors, output)
                y_view = Network(exchange.factors, output)
                cy = canonicalize(y_view, specs, fixed_output=True)
                if not cy.scalar:
                    continue
                y_key = cy.key if cy.key in ledger else None
                if y_key is None and not allow_bridge:
                    continue
                cz = canonicalize(Network(fused.factors, output), specs, fixed_output=True)
                if not cz.scalar:
                    continue
                item = _Candidate(x_key, y_key, cz.key, cz.network, cz.scalar,
                                  x_view, y_view, cx.scalar, cy.scalar,
                                  definition, (i, j), family)
                # One representative per signed cut orbit and outcome.
                signature = (frozenset((x_key, y_key)), cz.key,
                             cz.scalar * cx.scalar, definition.name)
                candidates.setdefault(signature, item)
        if interrupted:
            break
    return tuple(candidates.values()), interrupted


def _apply(ledger, candidate):
    x = ledger.get(candidate.x_key)
    y = ledger.get(candidate.y_key) if candidate.y_key is not None else None
    if x is None or (candidate.y_key is not None and y is None):
        return None
    a = x.coefficient * candidate.x_sign
    if candidate.x_key == candidate.y_key:
        # An exact residual symmetry may already imply Y = -X. The single
        # existing contribution a X becomes (a / 2) W, not two copies of X.
        if candidate.x_sign == candidate.y_sign:
            return None
        before = (WeightedNetwork(candidate.x_view, a),)
        after = (WeightedNetwork(candidate.z, (a * exact_scalar("1/2")) * candidate.z_sign),)
        return before, after
    b = y.coefficient * candidate.y_sign if y else 0
    before = (WeightedNetwork(candidate.x_view, a),)
    if y:
        before += (WeightedNetwork(candidate.y_view, b),)
    after = [WeightedNetwork(candidate.z, a * candidate.z_sign)]
    residual = a + b
    if residual:
        # Keep the actual exchanged view.  Its canonical key is obtained
        # below; no coefficient tolerance or absent-term completion is used.
        after.append(WeightedNetwork(candidate.y_view, residual))
    return before, tuple(after)


def _replace(ledger, candidate, specs):
    rewritten = _apply(ledger, candidate)
    if rewritten is None:
        return None
    before, after = rewritten
    result = dict(ledger)
    result.pop(candidate.x_key)
    if candidate.y_key is not None and candidate.y_key != candidate.x_key:
        result.pop(candidate.y_key)
    replacements = _collect(after, specs)
    for key, term in replacements.items():
        coefficient = term.coefficient + (result[key].coefficient if key in result else 0)
        if coefficient:
            result[key] = WeightedNetwork(term.network, coefficient)
        else:
            result.pop(key, None)
    step = RewriteStep(before, after, candidate.definition, candidate.positions)
    return result, step


def _branches(ledger, candidates, specs, definitions, max_types, large):
    """Batch compatible cuts on large targets instead of exploring every edge."""
    if not large:
        for candidate in candidates:
            replacement = _replace(ledger, candidate, specs)
            if replacement is not None:
                result, step = replacement
                yield result, (step,), candidate.y_key is None
        return
    # Favor exact opposite coefficients, but keep several different cut
    # seeds.  This is an explicitly restricted family, not an optimum proof.
    def priority(c):
        x = ledger[c.x_key].coefficient * c.x_sign
        y = ledger[c.y_key].coefficient * c.y_sign if c.y_key is not None else 0
        return (bool(x + y), c.y_key is None, repr(c.z_key), repr(c.x_key))
    ordered = sorted(candidates, key=priority)
    # A compute network can supply many transposed target views. Applying
    # unrelated cuts to those views destroys sharing and may increase calls.
    # Prefer a coherent cut across a whole compute orbit before mixing cuts.
    families = defaultdict(list)
    for candidate in ordered:
        families[candidate.family].append(candidate)
    ranked = sorted(families.values(),
                     key=lambda family: (sum(priority(c)[0] for c in family), -len(family),
                                         repr(family[0].family)))
    # Also assemble compatible *complete families*. Taking just one family
    # per search state wastes the budget on repeated whole-target scoring.
    # Keep each alternative family intact; never greedily cherry-pick a few
    # views from a conflicting alternative layout.
    layouts = []
    for seed in ranked[:4]:
        packed = []
        occupied = set()
        for family in (seed, *ranked):
            touched = {key for c in family for key in (c.x_key, c.y_key) if key is not None}
            if occupied.intersection(touched):
                continue
            packed.extend(family)
            occupied.update(touched)
        layouts.append(packed)
    layouts.extend(ranked[:4])
    yielded = set()
    for layout in layouts:
        result = dict(ledger)
        steps = []
        consumed = set()
        bridged = False
        single_type = next(iter(_used_types(result, definitions)), None) if max_types == 1 else None
        for candidate in layout:
            if candidate.x_key in consumed or candidate.y_key in consumed:
                continue
            if candidate.y_key is None and bridged:
                continue
            if max_types == 1:
                if single_type is not None and candidate.definition.name != single_type:
                    continue
            else:
                used = _used_types(result, definitions)
                if candidate.definition.name not in used and len(used) >= max_types:
                    continue
            replacement = _replace(result, candidate, specs)
            if replacement is None:
                continue
            result, step = replacement
            if max_types == 1:
                single_type = (candidate.definition.name if candidate.z_key in result else
                               next(iter(_used_types(result, definitions)), None))
            steps.append(step)
            consumed.add(candidate.x_key)
            if candidate.y_key is not None:
                consumed.add(candidate.y_key)
            bridged |= candidate.y_key is None
        key = _state_key(result)
        if steps and key not in yielded:
            yielded.add(key)
            yield result, tuple(steps), bridged


def block_reduce(basis_contract, basis, coefficients=None, *, tensor_specs,
                 max_w_types=1, search_budget=2000, search_seconds=5.0,
                 max_bridge_steps=1, pairing_family=None):
    """Plan fewer different full contractions of the normal Wick expression.

    ``search_budget`` limits expanded local states; ``search_seconds`` limits
    search after input normalization and optional family mapping.  The optional
    pairing family is symbolic data whose sum must equal the three-table target.
    Time/candidate truncation is reported,
    not called optimal.  One compensating ``X = RW + Y`` bridge is permitted
    by default, enabling multi-step reductions without inventing Wick terms.
    Its explicit residual is included in the certificate and whole-target
    score.  Set ``max_bridge_steps=0`` to search only existing partners.
    """
    with canonicalization_cache():
        token = _canonicalizers.set([])
        try:
            return _block_reduce(basis_contract, basis, coefficients, tensor_specs=tensor_specs,
                                 max_w_types=max_w_types, search_budget=search_budget,
                                 search_seconds=search_seconds, max_bridge_steps=max_bridge_steps,
                                 pairing_family=pairing_family)
        finally:
            _canonicalizers.reset(token)


def _block_reduce(basis_contract, basis, coefficients=None, *, tensor_specs,
                  max_w_types=1, search_budget=2000, search_seconds=5.0,
                  max_bridge_steps=1, pairing_family=None):
    if isinstance(max_w_types, bool) or not isinstance(max_w_types, int) or max_w_types < 1:
        raise ValueError("max_w_types must be a positive integer")
    if isinstance(search_budget, bool) or not isinstance(search_budget, int) or search_budget < 0:
        raise ValueError("search_budget must be a nonnegative integer")
    if search_seconds is not None and (isinstance(search_seconds, bool)
            or not isinstance(search_seconds, Real) or not isfinite(search_seconds)
            or search_seconds < 0):
        raise ValueError("search_seconds must be finite, nonnegative, or None")
    if isinstance(max_bridge_steps, bool) or not isinstance(max_bridge_steps, int) or max_bridge_steps < 0:
        raise ValueError("max_bridge_steps must be a nonnegative integer")
    start = monotonic()
    contracts, views, original_terms, input_specs = normalize_inputs(
        basis_contract, basis, coefficients, tensor_specs,
    )
    parse_seconds = monotonic() - start
    normalization_at = monotonic()
    _normal_only(input_specs)
    specs, definitions, by_propagator = _block_specs(input_specs)
    initial = _collect(original_terms, specs)
    free_cache = _BoundedCache(4096)
    cut_cache = _BoundedCache(2048)
    initial_score = _score(initial, specs, free_cache)
    normalization_seconds = monotonic() - normalization_at
    best = initial
    best_score = initial_score
    best_steps = ()
    best_family = None
    original_space = _output_space(next(iter(contracts.values())), input_specs) if contracts else ()
    if views:
        view = next(iter(views.values()))
        network = contracts[view.contract_id]
        original_space = _output_space(Network(network.factors,
                                  tuple(network.output[i] for i in view.permutation)), input_specs)
    family_mapping_seconds = 0.0
    if pairing_family is not None:
        from .global_layout import CompactPairingTerms, PairingFamily
        if not isinstance(pairing_family, PairingFamily):
            raise TypeError("pairing_family must be a PairingFamily")
        mapped_at = monotonic()
        raw_networks = ((pairing_family.terms.reference,)
                        if type(pairing_family.terms) is CompactPairingTerms
                        else pairing_family.terms)
        if any(f.tensor not in input_specs or input_specs[f.tensor].kind == "wedge"
               for t in raw_networks for f in t.network.factors):
            raise ValueError("the raw pairing family must contain only declared ordinary tensors")
        raw = _collect_family(pairing_family, specs)
        if _state_key(raw) != _state_key(initial):
            raise ValueError("the supplied pairing family differs from the weighted three-table target")
        family_mapping_seconds = monotonic() - mapped_at
    search_at = monotonic()
    deadline = float("inf") if search_seconds is None else search_at + search_seconds
    global_result = None
    global_fallback_reason = None
    if pairing_family is not None and search_budget and monotonic() < deadline:
        from .global_layout import find_pairing_layouts
        try:
            global_result = find_pairing_layouts(
                pairing_family, input_specs, max_w_types=max_w_types,
                search_seconds=max(0.0, deadline - monotonic()) if search_seconds is not None else 1e9,
                top_k=1,
            )
        except ValueError as error:
            # A certified sum need not have a uniform pairing template.
            # Its exact target remains eligible for ordinary local rewrites.
            global_fallback_reason = str(error)
        materialization_at = monotonic()
        for layout in (() if global_result is None else global_result.layouts[:1]):
            if layout.network_count >= initial_score[0]:
                break
            candidate_ledger = _collect(layout.fused_terms, specs)
            candidate_score = _score(candidate_ledger, specs, free_cache)
            if candidate_score[0] != layout.network_count:
                raise AssertionError("global orbit score disagrees with full target reconstruction")
            if candidate_score < best_score:
                best, best_score = candidate_ledger, candidate_score
                best_family = FamilyCertificate(
                    pairing_family.terms, layout.fused_terms,
                    layout.pairs, pairing_family.factor_positions,
                    tuple(definitions[name] for name in sorted({
                        factor.tensor for term in layout.fused_terms
                        for factor in term.network.factors if factor.tensor in definitions
                    })),
                )
                best_steps = ()
        materialization_seconds = monotonic() - materialization_at
    else:
        materialization_seconds = 0.0
    seed = best if best_family is not None else initial
    frontier = [(best_score if best_family else initial_score, 0, seed, (), 0)]
    visited = OrderedDict(((_state_key(seed), 0),))
    serial = 0
    expanded = 0
    truncated = False
    large = len(seed) > 24
    restricted = False
    achieved_lower_bound = not initial
    while frontier and expanded < search_budget and monotonic() < deadline:
        score, _serial, ledger, history, bridges = heapq.heappop(frontier)
        expanded += 1
        # Zero calls reaches a certified universal lower bound. One call
        # alone is not proof the target is nonzero: additional linear tensor
        # identities might still cancel its collection of output views.
        # Reserve part of each remaining time slice for scoring and applying
        # complete orbit batches, rather than spending the whole budget only
        # listing candidate cuts.
        now = monotonic()
        candidate_deadline = min(deadline, now + max(0.01, (deadline - now) / 2))
        candidates, interrupted = _candidates(
            ledger, specs, definitions, by_propagator, max_w_types,
            free_cache, candidate_deadline, bridges < max_bridge_steps, cut_cache,
        )
        truncated |= interrupted
        if not candidates:
            continue
        restricted |= large
        for child, steps, bridge in _branches(ledger, candidates, specs, definitions, max_w_types, large):
            bridge_count = bridges + int(bridge)
            key = _state_key(child)
            if key in visited and visited[key] <= bridge_count:
                visited.move_to_end(key)
                continue
            visited[key] = bridge_count
            visited.move_to_end(key)
            if large and len(visited) > 4096:
                visited.popitem(last=False)
                truncated = True
            child_score = _score(child, specs, free_cache)
            child_steps = history + steps
            if child_score < best_score:
                best, best_score, best_steps = child, child_score, child_steps
            serial += 1
            heapq.heappush(frontier, (child_score, serial, child, child_steps, bridge_count))
            if monotonic() >= deadline:
                truncated = True
                break
        if large and len(frontier) > 32:
            frontier = heapq.nsmallest(32, frontier)
            heapq.heapify(frontier)
            truncated = True
    search_elapsed_seconds = monotonic() - search_at - materialization_seconds
    exhausted = bool(frontier) or truncated or bool(global_result and not global_result.complete)
    complete = not exhausted and not restricted and pairing_family is None
    if best_score[0] == 0:
        achieved_lower_bound = True
    active = _used_types(best, definitions)
    networks, reconstruction = _reconstruct(best, specs, free_cache)
    plan = BlockPlan(
        tuple(definitions[name] for name in sorted(active)), networks, reconstruction,
        {name: spec for name, spec in specs.items() if spec.kind != "wedge" or name in active},
        original_terms, best_steps,
        {
            "input_contracts": len(contracts), "input_views": len(views),
            "initial_network_count": initial_score[0], "network_count": len(networks),
            "w_type_count": len(active), "search_states": expanded,
            "search_complete": complete, "budget_exhausted": exhausted,
            "optimality_scope": ("counting_lower_bound" if achieved_lower_bound else
                                  "restricted_rewrite_family" if complete else "best_found"),
            "restricted_batch_search": restricted,
            "search_strategy": "single_w_type" if max_w_types == 1 else "multi_w_type_generic",
            "output_shape": tuple(d for _type, d in original_space),
            "max_w_types": max_w_types, "max_bridge_steps": max_bridge_steps,
            "search_budget": search_budget, "search_seconds": search_seconds,
            "pairing_family_supplied": pairing_family is not None,
            "global_fallback_reason": global_fallback_reason,
            "global_layouts_tested": 0 if global_result is None else global_result.tested_layouts,
            "global_layout_complete": None if global_result is None else global_result.complete,
            "global_layout_scope": None if global_result is None else
                                   "supplied_pairing_family_global_matchings",
            "global_best_layouts": () if global_result is None else tuple(
                {"pairs": layout.pairs, "network_count": layout.network_count,
                 "burnside_count": None if layout.count is None else layout.count.orbit_count}
                for layout in global_result.layouts
            ),
            "auto_endpoint_generators": () if global_result is None else global_result.endpoint_generators,
            "auto_endpoint_witnesses": () if global_result is None else tuple(
                {"sink": witness.sink_permutation,
                 "source": witness.source_permutation,
                 "output": witness.output_permutation, "sign": witness.sign}
                for witness in global_result.endpoint_witnesses
            ),
            "normalization_seconds": normalization_seconds,
            "parse_seconds": parse_seconds,
            "family_mapping_seconds": family_mapping_seconds,
            "plan_materialization_seconds": materialization_seconds,
            "search_elapsed_seconds": search_elapsed_seconds,
            "symbolic_verified": False, "elapsed_seconds": monotonic() - start,
        },
        best_family,
    )
    proof_at = monotonic()
    # These ledgers were matched before the search.  Reuse that local proof
    # while checking the newly built certificate; public verify_plan always
    # recomputes its own target and family mapping.
    if not _verify_plan(plan, verified_target=initial,
                        verified_family_terms=(pairing_family.terms
                                               if pairing_family is not None else None)):
        raise AssertionError("block plan failed exact symbolic expansion verification")
    plan.stats["symbolic_verified"] = True
    plan.stats["proof_seconds"] = monotonic() - proof_at
    plan.stats["elapsed_seconds"] = monotonic() - start
    return plan


def _expand_network(network, definitions):
    """Yield the signed ordinary networks represented by fixed W factors."""
    for position, factor in enumerate(network.factors):
        if factor.tensor not in definitions:
            continue
        definition = definitions[factor.tensor]
        if len(factor.indices) != 8:
            raise ValueError("a W occurrence must expose four complete spin/color ports")
        a, b, i, j = [factor.indices[k:k + 2] for k in range(0, 8, 2)]
        def propagator(left, right):
            slots = [None] * 4
            for positions, labels in zip(definition.ports, (left, right)):
                for p, label in zip(positions, labels):
                    slots[p] = label
            return Factor(definition.propagator, tuple(slots))
        remaining = network.factors[:position] + network.factors[position + 1:]
        for sign, left_i, right_i in ((1, i, j), (-1, j, i)):
            ordinary = Network(remaining + (propagator(a, left_i), propagator(b, right_i)), network.output)
            for child_sign, child in _expand_network(ordinary, definitions):
                yield sign * child_sign, child
        return
    yield 1, network


def _expanded_sum(terms, definitions, specs):
    return _collect(
        (WeightedNetwork(network, term.coefficient * sign)
         for term in terms for sign, network in _expand_network(term.network, definitions)),
        specs,
    )


def verify_plan(plan):
    """Replay the proof and independently check its complete W expansion.

    Invalid or conflicting proof records fail closed. All comparisons use
    exact input coefficients, including the binary value of float inputs.
    """
    try:
        if _canonicalizers.get() is not None:
            return _verify_plan(plan)
        with canonicalization_cache():
            token = _canonicalizers.set([])
            try:
                return _verify_plan(plan)
            finally:
                _canonicalizers.reset(token)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False


def _verify_plan(plan, *, verified_target=None, verified_family_terms=None):
    definitions = {}
    for d in plan.w_definitions:
        if d.name in definitions:
            return False
        definitions[d.name] = d
    public_definitions = set(definitions)
    used_in_result = {factor.tensor for network in plan.networks
                      for factor in network.factors if factor.tensor in public_definitions}
    if used_in_result != public_definitions:
        return False
    # History can reference a type that later cancels out entirely.
    for step in plan.rewrites:
        if step.block.name in definitions and definitions[step.block.name] != step.block:
            return False
        definitions[step.block.name] = step.block
    if plan.family_certificate is not None:
        for definition in plan.family_certificate.w_definitions:
            if definition.name in definitions and definitions[definition.name] != definition:
                return False
            definitions[definition.name] = definition
    if any(f.tensor in definitions and f.tensor not in public_definitions
           for network in plan.networks for f in network.factors):
        return False
    specs = dict(plan.tensor_specs)
    if any(spec.kind == "wedge" and name not in definitions for name, spec in specs.items()):
        return False
    for d in definitions.values():
        source = specs[d.propagator]
        if source.kind != "peram" or source.mode != "normal" or d.ports != source.peram_ports:
            return False
        if d.shape != (3, 4) * 4:
            return False
        expected = TensorSpec(d.name, d.shape, ("color", "spin") * 4, kind="wedge",
            symmetries=(TensorSymmetry((2, 3, 0, 1, 4, 5, 6, 7), -1),
                        TensorSymmetry((0, 1, 2, 3, 6, 7, 4, 5), -1)))
        if d.name in specs and specs[d.name] != expected:
            return False
        specs[d.name] = expected
    _normal_only({name: spec for name, spec in specs.items() if spec.kind != "wedge"})
    limit = plan.stats.get("max_w_types")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                              or limit < 1 or len(plan.w_definitions) > limit):
        return False
    if any(f.tensor in definitions for t in plan.original_terms for f in t.network.factors):
        return False
    target = (verified_target if verified_target is not None
              else _collect(plan.original_terms, specs))
    current = target
    if plan.family_certificate is not None:
        family = plan.family_certificate
        if (not family.raw_terms or not family.fused_terms
                or len(set(family.factor_positions)) != len(family.factor_positions)):
            return False
        used_slots = tuple(slot for pair in family.pairs for slot in pair)
        if (len(set(used_slots)) != len(used_slots)
                or any(type(slot) is not int or slot < 0 or slot >= len(family.factor_positions)
                       for slot in used_slots)):
            return False
        if any(len(pair) != 2 for pair in family.pairs):
            return False
        from .global_layout import CompactPairingTerms
        checked_terms = ((family.raw_terms.reference,)
                         if type(family.raw_terms) is CompactPairingTerms
                         else family.raw_terms)
        for term in checked_terms:
            if any(f.tensor in definitions for f in term.network.factors):
                return False
            factors = term.network.factors
            if any(type(i) is not int or i < 0 or i >= len(factors)
                   or specs[factors[i].tensor].kind != "peram" for i in family.factor_positions):
                return False
            for left, right in family.pairs:
                a, b = (specs[factors[family.factor_positions[i]].tensor] for i in (left, right))
                if (a.identity or a.name) != (b.identity or b.name) or a.peram_ports != b.peram_ports:
                    return False
        if family.raw_terms is verified_family_terms:
            raw = target
        else:
            from .global_layout import PairingFamily
            raw = _collect_family(PairingFamily(family.raw_terms, family.factor_positions), specs)
            if _state_key(raw) != _state_key(target):
                return False
        current = _collect(family.fused_terms, specs)
        # The placement metadata is part of the proof, not an unchecked
        # annotation. Rebuild its signed exchange orbits independently.
        from .global_layout import PairingFamily, _prepare, _orbit, _fused
        ordinary_specs = {name: spec for name, spec in specs.items() if spec.kind != "wedge"}
        # An override of compact iteration or coefficient methods is an
        # explicit family, not an instance of the fixed alternating law.
        raw_family_terms = (tuple(family.raw_terms)
                            if isinstance(family.raw_terms, CompactPairingTerms)
                            and type(family.raw_terms) is not CompactPairingTerms
                            else family.raw_terms)
        prepared = _prepare(PairingFamily(raw_family_terms, family.factor_positions), ordinary_specs)
        pairings, _sink, _source, allowed, *_rest = prepared
        if any(tuple(sorted(pair)) not in allowed for pair in family.pairs):
            return False
        by_propagator = {}
        for name, spec in ordinary_specs.items():
            if spec.kind == "peram":
                for definition in definitions.values():
                    source = ordinary_specs[definition.propagator]
                    if (source.identity or source.name) == (spec.identity or spec.name):
                        by_propagator[name] = definition
        compact = type(family.raw_terms) is CompactPairingTerms
        if compact and any(left >= right for left, right in family.pairs):
            # CompactPairingTerms.representatives relies on this orientation
            # to choose exactly one member of every exchange orbit.
            return False
        representatives = (family.raw_terms.representatives(family.pairs) if compact
                           else sorted(pairings))
        expected_fused, seen = [], set()
        for pairing in representatives:
            if not compact and pairing in seen:
                continue
            term = pairings[pairing]
            if not compact:
                orbit = _orbit(pairing, family.pairs)
                if any(image not in pairings or pairings[image].coefficient !=
                       term.coefficient * (-1) ** bin(mask).count("1")
                       for mask, image in enumerate(orbit)):
                    return False
                seen.update(orbit)
            expected_fused.append(WeightedNetwork(
                _fused(term.network, family.pairs, family.factor_positions, specs, by_propagator),
                term.coefficient,
            ))
        if _state_key(_collect(expected_fused, specs)) != _state_key(current):
            return False
        # A compact family contains every class-preserving bijection exactly
        # once, with its exact permutation-parity coefficient.  Disjoint
        # legal pairs partition it into 2**len(pairs) exchange orbits; the
        # signed expansion of each fixed W reproduces precisely one orbit.
        # The checked expected-fused ledger thus proves the complete raw sum
        # without materializing its W expansion.  Explicit families retain
        # the independent expansion check.
        if not compact and _state_key(_expanded_sum(family.fused_terms, definitions, specs)) != _state_key(raw):
            return False
    for step in plan.rewrites:
        if not step.before:
            return False
        positions = step.factor_positions
        factors = step.before[0].network.factors
        if (len(positions) != 2 or len(set(positions)) != 2
                or any(type(i) is not int or not 0 <= i < len(factors) for i in positions)):
            return False
        source = specs[step.block.propagator]
        identity = source.name if source.identity is None else source.identity
        for i in positions:
            candidate = specs[factors[i].tensor]
            if (candidate.kind != "peram" or candidate.peram_ports != step.block.ports
                    or (candidate.name if candidate.identity is None else candidate.identity) != identity):
                return False
        before = _expanded_sum(step.before, definitions, specs)
        after = _expanded_sum(step.after, definitions, specs)
        if _state_key(before) != _state_key(after):
            return False
        consumed = _collect(step.before, specs)
        if any(key not in current or current[key].coefficient != term.coefficient
               for key, term in consumed.items()):
            return False
        for key in consumed:
            del current[key]
        for key, term in _collect(step.after, specs).items():
            coefficient = term.coefficient + (current[key].coefficient if key in current else 0)
            if coefficient:
                current[key] = WeightedNetwork(term.network, coefficient)
            else:
                current.pop(key, None)
    reconstructed = []
    for term in plan.reconstruction:
        if term.network_id < 0 or term.network_id >= len(plan.networks):
            return False
        network = plan.networks[term.network_id]
        if sorted(term.permutation) != list(range(len(network.output))):
            return False
        viewed = Network(network.factors, tuple(network.output[i] for i in term.permutation))
        reconstructed.append(WeightedNetwork(viewed, term.coefficient))
    # Each local rewrite (or global family replacement) was already checked
    # after expanding W, and the current ledger reconstructs exactly. Their
    # transitive equality proves the full target without a second expansion.
    return _state_key(current) == _state_key(_collect(reconstructed, specs))
