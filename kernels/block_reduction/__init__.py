"""Independent, transpose-aware antisymmetric block reduction of tensor sums."""
from .model import (
    BasisView, BlockDefinition, BlockPlan, FamilyCertificate, Factor, Network, ReconstructionTerm,
    RewriteStep, TensorSpec, TensorSymmetry, WeightedNetwork, compose_transposes,
)
from .planner import block_reduce, verify_plan
from .evaluate import evaluate_expression, evaluate_plan
from .io import SymbolicInput, load_symbolic_output
from .global_layout import (CompactPairingTerms, PairingFamily, LayoutResult,
                            EndpointWitness, GlobalLayoutResult, find_pairing_layouts)

__all__ = [
    "block_reduce", "verify_plan", "BasisView", "BlockDefinition", "BlockPlan", "FamilyCertificate",
    "Factor", "Network", "ReconstructionTerm", "RewriteStep", "TensorSpec",
    "TensorSymmetry", "WeightedNetwork", "compose_transposes", "evaluate_expression",
    "evaluate_plan", "SymbolicInput", "load_symbolic_output",
    "PairingFamily", "CompactPairingTerms", "LayoutResult", "EndpointWitness",
    "GlobalLayoutResult", "find_pairing_layouts",
]
