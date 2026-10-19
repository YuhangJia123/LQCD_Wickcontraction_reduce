"""Seven ordinary-colour correlators built through the public input API."""

from __future__ import annotations

from .user_api import HadronOperator, make_operator


def seven_examples() -> dict[str, tuple[list[HadronOperator], list[HadronOperator]]]:
    """Return the seven established meson/baryon examples as [sink, source]."""
    def baryon(side: str, species: str, label: str, flavors, external_spin=0):
        gamma = "Cgamma3" if species.startswith("Delta") else "Cgamma5"
        return make_operator(
            side, "baryon", species, label, flavors, gamma,
            gamma_slots=(0, 1), open_slot=2,
            external_spin=external_spin if side == "source" else None,
        )

    def pion(side: str, label: str):
        return make_operator(
            side, "meson", "pi0_uubar", label, ("u",), "gamma5"
        )

    p = ("u", "d", "u")
    n = ("d", "u", "d")
    delta_pp = ("u", "u", "u")
    delta_mm = ("d", "d", "d")

    ppn = (
        [baryon("sink", "p", "sink_p1", p),
         baryon("sink", "p", "sink_p2", p),
         baryon("sink", "n", "sink_n", n)],
        [baryon("source", "p", "source_p1", p, 0),
         baryon("source", "p", "source_p2", p, 1),
         baryon("source", "n", "source_n", n, 1)],
    )
    return {
        "pipi": (
            [pion("sink", "sink_pi1"), pion("sink", "sink_pi2")],
            [pion("source", "source_pi1"), pion("source", "source_pi2")],
        ),
        "pn": (
            [baryon("sink", "p", "sink_p", p),
             baryon("sink", "n", "sink_n", n)],
            [baryon("source", "p", "source_p", p, 0),
             baryon("source", "n", "source_n", n, 1)],
        ),
        "ppn": ppn,
        "delta": (
            [baryon("sink", "Delta++", "sink_delta", delta_pp)],
            [baryon("source", "Delta++", "source_delta", delta_pp, 0)],
        ),
        "delta++delta-": (
            [baryon("sink", "Delta++", "sink_delta_pp", delta_pp),
             baryon("sink", "Delta-", "sink_delta_mm", delta_mm)],
            [baryon("source", "Delta++", "source_delta_pp", delta_pp, 0),
             baryon("source", "Delta-", "source_delta_mm", delta_mm, 0)],
        ),
        "pip": (
            [pion("sink", "sink_pi"), baryon("sink", "p", "sink_p", p)],
            [pion("source", "source_pi"),
             baryon("source", "p", "source_p", p, 0)],
        ),
        "pip-p": (
            [baryon("sink", "p", "sink_p", p)],
            [pion("source", "source_pi"),
             baryon("source", "p", "source_p", p, 0)],
        ),
    }


__all__ = ["seven_examples"]
