"""Explicit scalar `contract` expressions for the reconstructed ppn 93 blocks.

The input arrays use the same shapes and conventions as the ppn operator
fixture in this repository. Each `block_values[r]` is a signed sum of the
eight Wick terms in one representative projected block; `block_weights[r]`
is its exact proton-exchange orbit size (2 or 4). This is a reconstruction
for the stated ppn projection, not a published survivor list from the paper.
"""

from __future__ import annotations

import numpy as np
from opt_einsum import contract


def ppn_projected_93(
    C_g5, C_g5_dagger, peram_u_sin_sou, VVV_sink, conj_VVV_source,
    P_ppn=None,
):
    """Evaluate the 93 weighted ppn blocks as scalar index contractions."""
    if P_ppn is None:
        psi = np.zeros((2, 2, 2), dtype=np.complex128)
        psi[0, 1, 0] = 1.0 / np.sqrt(6.0)
        psi[1, 0, 0] = -1.0 / np.sqrt(6.0)
        spin_projector = np.einsum("abc,def->abcdef", psi, psi.conjugate())
        P_ppn = np.zeros((4, 4, 4, 4, 4, 4), dtype=np.complex128)
        P_ppn[:2, :2, :2, :2, :2, :2] = spin_projector
    projected_operands = (
        C_g5, C_g5, C_g5, C_g5_dagger, C_g5_dagger, C_g5_dagger,
        peram_u_sin_sou, peram_u_sin_sou, peram_u_sin_sou,
        peram_u_sin_sou, peram_u_sin_sou, peram_u_sin_sou,
        peram_u_sin_sou, peram_u_sin_sou, peram_u_sin_sou,
        VVV_sink, VVV_sink, VVV_sink,
        conj_VVV_source, conj_VVV_source, conj_VVV_source,
        P_ppn,
    )
    block_values = []
    # Orbit 0: representative state Wick ID 1; weight 2.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqGj,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqyE,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 3
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqGj,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 25
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqyE,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 27
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqGj,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 121
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqyE,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 123
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqGj,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 145
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqyE,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 147
    )

    # Orbit 1: representative state Wick ID 2; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqIk,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqyE,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 5
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqIk,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 26
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqyE,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 29
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqIk,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 122
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqyE,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 125
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqIk,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 146
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqyE,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 149
    )

    # Orbit 2: representative state Wick ID 4; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqIk,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 4
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqGj,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 6
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqIk,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 28
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqGj,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 30
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqIk,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 124
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqGj,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 126
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqIk,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 148
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqGj,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 150
    )

    # Orbit 3: representative state Wick ID 7; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqGj,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 7
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqxh,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 9
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqGj,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 49
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqxh,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 51
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqGj,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 127
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqxh,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 129
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqGj,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 169
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqxh,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 171
    )

    # Orbit 4: representative state Wick ID 8; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqIk,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 8
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqxh,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 11
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqIk,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 50
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqxh,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 53
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqIk,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 128
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqxh,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 131
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqIk,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 170
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqxh,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 173
    )

    # Orbit 5: representative state Wick ID 10; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqIk,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 10
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqGj,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 12
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqIk,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 52
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqGj,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 54
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqIk,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 130
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqGj,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 132
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqIk,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 172
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqGj,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 174
    )

    # Orbit 6: representative state Wick ID 13; weight 2.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqyE,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 13
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqxh,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 15
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqyE,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 73
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqxh,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 75
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqyE,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 133
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqxh,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 135
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqyE,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 193
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqxh,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 195
    )

    # Orbit 7: representative state Wick ID 14; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqIk,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 14
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqxh,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 17
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqIk,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 74
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqxh,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 77
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqIk,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 134
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqxh,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 137
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqIk,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 194
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqxh,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 197
    )

    # Orbit 8: representative state Wick ID 16; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqIk,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 16
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqyE,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 18
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqIk,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 76
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqyE,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 78
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqIk,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 136
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqyE,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 138
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqIk,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 196
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqyE,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 198
    )

    # Orbit 9: representative state Wick ID 20; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,Bpxh,cqGj,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 20
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpGj,cqxh,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 23
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,Bpxh,cqGj,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 98
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpGj,cqxh,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 101
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,Bpxh,cqGj,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 140
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpGj,cqxh,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 143
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,Bpxh,cqGj,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 218
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpGj,cqxh,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 221
    )

    # Orbit 10: representative state Wick ID 31; weight 2.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqGj,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 31
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqvD,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 33
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqGj,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 55
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqvD,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 57
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqGj,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 151
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqvD,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 153
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqGj,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 175
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqvD,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 177
    )

    # Orbit 11: representative state Wick ID 32; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqIk,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 32
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqvD,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 35
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqIk,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 56
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqvD,drzi,CsHF,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 59
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqIk,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 152
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqvD,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 155
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqIk,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 176
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqvD,drzi,CsJl,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 179
    )

    # Orbit 12: representative state Wick ID 34; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqIk,drzi,CsHF,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 34
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqGj,drzi,CsHF,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 36
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqIk,drzi,CsHF,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 58
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqGj,drzi,CsHF,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 60
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqIk,drzi,CsJl,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 154
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqGj,drzi,CsJl,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 156
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqIk,drzi,CsJl,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 178
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqGj,drzi,CsJl,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 180
    )

    # Orbit 13: representative state Wick ID 38; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpvD,cqIk,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 38
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpIk,cqvD,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 41
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpvD,cqIk,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 80
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpIk,cqvD,drzi,CsHF,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 83
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpvD,cqIk,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 158
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpIk,cqvD,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 161
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpvD,cqIk,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 200
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpIk,cqvD,drzi,CsJl,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 203
    )

    # Orbit 14: representative state Wick ID 62; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqIk,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 62
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqvD,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 65
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqIk,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 86
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqvD,drzi,CsHF,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 89
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqIk,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 182
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqvD,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 185
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqIk,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 206
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqvD,drzi,CsJl,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 209
    )

    # Orbit 15: representative state Wick ID 61; weight 2.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqxh,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 61
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqvD,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 63
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqxh,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 85
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqvD,drzi,CsHF,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 87
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqxh,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 181
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqvD,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 183
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqxh,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 205
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqvD,drzi,CsJl,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 207
    )

    # Orbit 16: representative state Wick ID 64; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqIk,drzi,CsHF,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 64
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqxh,drzi,CsHF,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 66
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqIk,drzi,CsHF,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 88
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqxh,drzi,CsHF,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 90
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqIk,drzi,CsJl,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 184
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqxh,drzi,CsJl,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 186
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqIk,drzi,CsJl,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 208
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqxh,drzi,CsJl,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 210
    )

    # Orbit 17: representative state Wick ID 241; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqGj,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 241
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqyE,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 243
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqGj,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 265
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqyE,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 267
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqGj,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 361
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqyE,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 363
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqGj,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 385
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqyE,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 387
    )

    # Orbit 18: representative state Wick ID 242; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqIk,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 242
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqyE,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 245
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqIk,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 266
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqyE,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 269
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqIk,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 362
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqyE,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 365
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqIk,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 386
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqyE,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 389
    )

    # Orbit 19: representative state Wick ID 244; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqIk,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 244
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqGj,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 246
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqIk,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 268
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqGj,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 270
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqIk,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 364
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqGj,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 366
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqIk,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 388
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqGj,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 390
    )

    # Orbit 20: representative state Wick ID 247; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqGj,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 247
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqxh,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 249
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqGj,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 289
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqxh,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 291
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqGj,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 367
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqxh,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 369
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqGj,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 409
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqxh,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 411
    )

    # Orbit 21: representative state Wick ID 248; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqIk,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 248
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqxh,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 251
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqIk,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 290
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqxh,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 293
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqIk,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 368
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqxh,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 371
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqIk,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 410
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqxh,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 413
    )

    # Orbit 22: representative state Wick ID 250; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqIk,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 250
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqGj,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 252
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqIk,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 292
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqGj,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 294
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqIk,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 370
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqGj,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 372
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqIk,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 412
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqGj,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 414
    )

    # Orbit 23: representative state Wick ID 253; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqyE,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 253
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqxh,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 255
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqyE,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 313
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqxh,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 315
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqyE,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 373
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqxh,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 375
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqyE,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 433
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqxh,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 435
    )

    # Orbit 24: representative state Wick ID 254; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqIk,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 254
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqxh,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 257
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqIk,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 314
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqxh,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 317
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqIk,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 374
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqxh,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 377
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqIk,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 434
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqxh,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 437
    )

    # Orbit 25: representative state Wick ID 256; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqIk,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 256
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqyE,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 258
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqIk,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 316
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqyE,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 318
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqIk,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 376
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqyE,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 378
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqIk,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 436
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqyE,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 438
    )

    # Orbit 26: representative state Wick ID 259; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,Bpxh,cqyE,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 259
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpyE,cqxh,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 261
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,Bpxh,cqyE,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 337
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpyE,cqxh,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 339
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,Bpxh,cqyE,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 379
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpyE,cqxh,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 381
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,Bpxh,cqyE,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 457
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpyE,cqxh,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 459
    )

    # Orbit 27: representative state Wick ID 260; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,Bpxh,cqGj,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 260
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpGj,cqxh,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 263
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,Bpxh,cqGj,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 338
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpGj,cqxh,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 341
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,Bpxh,cqGj,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 380
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpGj,cqxh,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 383
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,Bpxh,cqGj,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 458
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpGj,cqxh,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 461
    )

    # Orbit 28: representative state Wick ID 262; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpyE,cqGj,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 262
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpGj,cqyE,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 264
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpyE,cqGj,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 340
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpGj,cqyE,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 342
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpyE,cqGj,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 382
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpGj,cqyE,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 384
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpyE,cqGj,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 460
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpGj,cqyE,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 462
    )

    # Orbit 29: representative state Wick ID 271; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqGj,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 271
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqvD,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 273
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqGj,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 295
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqvD,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 297
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqGj,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 391
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqvD,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 393
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqGj,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 415
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqvD,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 417
    )

    # Orbit 30: representative state Wick ID 272; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqIk,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 272
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqvD,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 275
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqIk,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 296
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqvD,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 299
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqIk,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 392
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqvD,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 395
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqIk,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 416
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqvD,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 419
    )

    # Orbit 31: representative state Wick ID 274; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqIk,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 274
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqGj,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 276
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqIk,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 298
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqGj,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 300
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqIk,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 394
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqGj,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 396
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqIk,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 418
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqGj,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 420
    )

    # Orbit 32: representative state Wick ID 277; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpvD,cqyE,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 277
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpyE,cqvD,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 279
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpvD,cqyE,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 319
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpyE,cqvD,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 321
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpvD,cqyE,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 397
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpyE,cqvD,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 399
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpvD,cqyE,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 439
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpyE,cqvD,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 441
    )

    # Orbit 33: representative state Wick ID 278; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpvD,cqIk,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 278
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpIk,cqvD,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 281
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpvD,cqIk,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 320
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpIk,cqvD,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 323
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpvD,cqIk,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 398
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpIk,cqvD,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 401
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpvD,cqIk,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 440
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpIk,cqvD,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 443
    )

    # Orbit 34: representative state Wick ID 280; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpyE,cqIk,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 280
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpIk,cqyE,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 282
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpyE,cqIk,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 322
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpIk,cqyE,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 324
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpyE,cqIk,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 400
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpIk,cqyE,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 402
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpyE,cqIk,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 442
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpIk,cqyE,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 444
    )

    # Orbit 35: representative state Wick ID 283; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpvD,cqyE,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 283
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpyE,cqvD,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 285
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpvD,cqyE,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 343
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpyE,cqvD,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 345
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpvD,cqyE,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 403
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpyE,cqvD,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 405
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpvD,cqyE,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 463
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpyE,cqvD,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 465
    )

    # Orbit 36: representative state Wick ID 284; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpvD,cqGj,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 284
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpGj,cqvD,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 287
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpvD,cqGj,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 344
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpGj,cqvD,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 347
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpvD,cqGj,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 404
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpGj,cqvD,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 407
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpvD,cqGj,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 464
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpGj,cqvD,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 467
    )

    # Orbit 37: representative state Wick ID 286; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpyE,cqGj,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 286
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpGj,cqyE,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 288
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpyE,cqGj,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 346
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpGj,cqyE,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 348
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpyE,cqGj,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 406
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpGj,cqyE,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 408
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpyE,cqGj,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 466
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpGj,cqyE,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 468
    )

    # Orbit 38: representative state Wick ID 301; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqxh,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 301
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqvD,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 303
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqxh,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 325
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqvD,drHF,Cszi,etJl,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 327
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqxh,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 421
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqvD,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 423
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqxh,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 445
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqvD,drHF,CsJl,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 447
    )

    # Orbit 39: representative state Wick ID 302; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqIk,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 302
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqvD,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 305
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqIk,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 326
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqvD,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 329
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqIk,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 422
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqvD,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 425
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqIk,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 446
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqvD,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 449
    )

    # Orbit 40: representative state Wick ID 304; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqIk,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 304
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqxh,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 306
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqIk,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 328
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqxh,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 330
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqIk,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 424
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqxh,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 426
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqIk,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 448
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqxh,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 450
    )

    # Orbit 41: representative state Wick ID 307; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpvD,cqxh,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 307
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,Bpxh,cqvD,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 309
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpvD,cqxh,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 349
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,Bpxh,cqvD,drHF,Cszi,etJl,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 351
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpvD,cqxh,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 427
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,Bpxh,cqvD,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 429
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpvD,cqxh,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 469
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,Bpxh,cqvD,drHF,CsJl,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 471
    )

    # Orbit 42: representative state Wick ID 308; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpvD,cqGj,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 308
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpGj,cqvD,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 311
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpvD,cqGj,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 350
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpGj,cqvD,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 353
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpvD,cqGj,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 428
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpGj,cqvD,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 431
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpvD,cqGj,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 470
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpGj,cqvD,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 473
    )

    # Orbit 43: representative state Wick ID 310; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,Bpxh,cqGj,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 310
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpGj,cqxh,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 312
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,Bpxh,cqGj,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 352
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpGj,cqxh,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 354
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,Bpxh,cqGj,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 430
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpGj,cqxh,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 432
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,Bpxh,cqGj,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 472
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpGj,cqxh,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 474
    )

    # Orbit 44: representative state Wick ID 331; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpvD,cqxh,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 331
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,Bpxh,cqvD,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 333
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpvD,cqxh,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 355
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,Bpxh,cqvD,drHF,Cszi,etJl,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 357
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpvD,cqxh,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 451
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,Bpxh,cqvD,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 453
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpvD,cqxh,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 475
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,Bpxh,cqvD,drHF,CsJl,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 477
    )

    # Orbit 45: representative state Wick ID 332; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpvD,cqyE,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 332
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpyE,cqvD,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 335
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpvD,cqyE,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 356
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpyE,cqvD,drHF,Cszi,etJl,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 359
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpvD,cqyE,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 452
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpyE,cqvD,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 455
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpvD,cqyE,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 476
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpyE,cqvD,drHF,CsJl,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 479
    )

    # Orbit 46: representative state Wick ID 334; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,Bpxh,cqyE,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 334
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpyE,cqxh,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 336
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,Bpxh,cqyE,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 358
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpyE,cqxh,drHF,Cszi,etJl,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 360
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,Bpxh,cqyE,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 454
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpyE,cqxh,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 456
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,Bpxh,cqyE,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 478
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpyE,cqxh,drHF,CsJl,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 480
    )

    # Orbit 47: representative state Wick ID 481; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqGj,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 481
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqyE,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 483
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqGj,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 505
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqyE,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 507
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqGj,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 601
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqyE,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 603
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqGj,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 625
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqyE,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 627
    )

    # Orbit 48: representative state Wick ID 482; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqIk,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 482
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqyE,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 485
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqIk,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 506
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqyE,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 509
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpyE,cqIk,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 602
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqyE,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 605
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpyE,cqIk,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 626
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqyE,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 629
    )

    # Orbit 49: representative state Wick ID 484; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqIk,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 484
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqGj,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 486
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqIk,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 508
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqGj,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 510
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpGj,cqIk,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 604
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,bowg,BpIk,cqGj,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 606
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpGj,cqIk,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 628
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,bowg,BpIk,cqGj,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 630
    )

    # Orbit 50: representative state Wick ID 487; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqGj,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 487
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqxh,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 489
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqGj,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 529
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqxh,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 531
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqGj,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 607
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqxh,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 609
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqGj,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 649
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqxh,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 651
    )

    # Orbit 51: representative state Wick ID 488; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqIk,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 488
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqxh,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 491
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqIk,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 530
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqxh,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 533
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,Bpxh,cqIk,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 608
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqxh,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 611
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,Bpxh,cqIk,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 650
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqxh,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 653
    )

    # Orbit 52: representative state Wick ID 490; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqIk,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 490
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqGj,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 492
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqIk,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 532
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqGj,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 534
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpGj,cqIk,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 610
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,bowg,BpIk,cqGj,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 612
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpGj,cqIk,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 652
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,bowg,BpIk,cqGj,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 654
    )

    # Orbit 53: representative state Wick ID 493; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqyE,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 493
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqxh,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 495
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqyE,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 553
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqxh,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 555
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqyE,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 613
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqxh,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 615
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqyE,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 673
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqxh,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 675
    )

    # Orbit 54: representative state Wick ID 494; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqIk,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 494
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqxh,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 497
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqIk,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 554
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqxh,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 557
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,Bpxh,cqIk,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 614
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqxh,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 617
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,Bpxh,cqIk,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 674
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqxh,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 677
    )

    # Orbit 55: representative state Wick ID 496; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqIk,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 496
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqyE,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 498
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqIk,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 556
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqyE,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 558
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpyE,cqIk,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 616
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,bowg,BpIk,cqyE,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 618
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpyE,cqIk,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 676
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,bowg,BpIk,cqyE,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 678
    )

    # Orbit 56: representative state Wick ID 499; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,Bpxh,cqyE,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 499
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpyE,cqxh,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 501
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,Bpxh,cqyE,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 577
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpyE,cqxh,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 579
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,Bpxh,cqyE,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 619
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpyE,cqxh,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 621
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,Bpxh,cqyE,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 697
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpyE,cqxh,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 699
    )

    # Orbit 57: representative state Wick ID 500; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,Bpxh,cqGj,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 500
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpGj,cqxh,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 503
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,Bpxh,cqGj,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 578
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpGj,cqxh,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 581
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,Bpxh,cqGj,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 620
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpGj,cqxh,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 623
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,Bpxh,cqGj,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 698
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpGj,cqxh,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 701
    )

    # Orbit 58: representative state Wick ID 502; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpyE,cqGj,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 502
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpGj,cqyE,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 504
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpyE,cqGj,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 580
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpGj,cqyE,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 582
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpyE,cqGj,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 622
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,bowg,BpGj,cqyE,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 624
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpyE,cqGj,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 700
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,bowg,BpGj,cqyE,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 702
    )

    # Orbit 59: representative state Wick ID 511; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqGj,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 511
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqvD,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 513
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqGj,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 535
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqvD,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 537
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqGj,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 631
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqvD,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 633
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqGj,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 655
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqvD,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 657
    )

    # Orbit 60: representative state Wick ID 512; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqIk,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 512
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqvD,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 515
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqIk,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 536
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqvD,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 539
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpvD,cqIk,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 632
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqvD,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 635
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpvD,cqIk,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 656
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqvD,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 659
    )

    # Orbit 61: representative state Wick ID 514; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqIk,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 514
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqGj,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 516
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqIk,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 538
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqGj,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 540
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpGj,cqIk,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 634
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anyE,bowg,BpIk,cqGj,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 636
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpGj,cqIk,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 658
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anxh,bowg,BpIk,cqGj,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 660
    )

    # Orbit 62: representative state Wick ID 517; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpvD,cqyE,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 517
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpyE,cqvD,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 519
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpvD,cqyE,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 559
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpyE,cqvD,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 561
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpvD,cqyE,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 637
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpyE,cqvD,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 639
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpvD,cqyE,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 679
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpyE,cqvD,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 681
    )

    # Orbit 63: representative state Wick ID 518; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpvD,cqIk,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 518
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpIk,cqvD,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 521
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpvD,cqIk,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 560
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpIk,cqvD,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 563
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpvD,cqIk,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 638
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpIk,cqvD,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 641
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpvD,cqIk,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 680
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpIk,cqvD,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 683
    )

    # Orbit 64: representative state Wick ID 520; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpyE,cqIk,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 520
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpIk,cqyE,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 522
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpyE,cqIk,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 562
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpIk,cqyE,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 564
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpyE,cqIk,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 640
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,bowg,BpIk,cqyE,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 642
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpyE,cqIk,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 682
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,bowg,BpIk,cqyE,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 684
    )

    # Orbit 65: representative state Wick ID 523; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpvD,cqyE,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 523
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpyE,cqvD,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 525
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpvD,cqyE,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 583
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpyE,cqvD,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 585
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpvD,cqyE,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 643
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpyE,cqvD,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 645
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpvD,cqyE,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 703
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpyE,cqvD,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 705
    )

    # Orbit 66: representative state Wick ID 524; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpvD,cqGj,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 524
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpGj,cqvD,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 527
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpvD,cqGj,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 584
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpGj,cqvD,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 587
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpvD,cqGj,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 644
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpGj,cqvD,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 647
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpvD,cqGj,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 704
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpGj,cqvD,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 707
    )

    # Orbit 67: representative state Wick ID 526; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpyE,cqGj,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 526
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpGj,cqyE,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 528
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpyE,cqGj,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 586
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpGj,cqyE,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 588
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpyE,cqGj,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 646
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,bowg,BpGj,cqyE,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 648
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpyE,cqGj,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 706
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,bowg,BpGj,cqyE,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 708
    )

    # Orbit 68: representative state Wick ID 541; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqxh,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 541
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqvD,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 543
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqxh,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 565
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqvD,drJl,Cszi,etHF,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 567
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqxh,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 661
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqvD,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 663
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqxh,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 685
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqvD,drJl,CsHF,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 687
    )

    # Orbit 69: representative state Wick ID 542; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqIk,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 542
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqvD,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 545
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqIk,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 566
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqvD,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 569
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpvD,cqIk,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 662
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqvD,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 665
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpvD,cqIk,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 686
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqvD,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 689
    )

    # Orbit 70: representative state Wick ID 544; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqIk,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 544
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqxh,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 546
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqIk,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 568
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqxh,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 570
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,Bpxh,cqIk,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 664
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anGj,bowg,BpIk,cqxh,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 666
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,Bpxh,cqIk,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 688
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anyE,bowg,BpIk,cqxh,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 690
    )

    # Orbit 71: representative state Wick ID 548; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpvD,cqGj,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 548
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpGj,cqvD,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 551
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpvD,cqGj,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 590
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpGj,cqvD,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 593
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpvD,cqGj,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 668
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpGj,cqvD,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 671
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpvD,cqGj,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 710
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpGj,cqvD,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 713
    )

    # Orbit 72: representative state Wick ID 547; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpvD,cqxh,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 547
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,Bpxh,cqvD,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 549
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpvD,cqxh,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 589
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,Bpxh,cqvD,drJl,Cszi,etHF,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 591
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpvD,cqxh,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 667
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,Bpxh,cqvD,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 669
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpvD,cqxh,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 709
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,Bpxh,cqvD,drJl,CsHF,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 711
    )

    # Orbit 73: representative state Wick ID 550; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,Bpxh,cqGj,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 550
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpGj,cqxh,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 552
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,Bpxh,cqGj,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 592
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpGj,cqxh,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 594
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,Bpxh,cqGj,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 670
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anIk,bowg,BpGj,cqxh,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 672
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,Bpxh,cqGj,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 712
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anyE,bowg,BpGj,cqxh,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 714
    )

    # Orbit 74: representative state Wick ID 572; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpvD,cqyE,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 572
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpyE,cqvD,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 575
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpvD,cqyE,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 596
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpyE,cqvD,drJl,Cszi,etHF,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 599
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpvD,cqyE,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 692
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpyE,cqvD,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 695
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpvD,cqyE,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 716
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpyE,cqvD,drJl,CsHF,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 719
    )

    # Orbit 75: representative state Wick ID 571; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpvD,cqxh,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 571
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,Bpxh,cqvD,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 573
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpvD,cqxh,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 595
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,Bpxh,cqvD,drJl,Cszi,etHF,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 597
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpvD,cqxh,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 691
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,Bpxh,cqvD,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 693
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpvD,cqxh,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 715
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,Bpxh,cqvD,drJl,CsHF,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 717
    )

    # Orbit 76: representative state Wick ID 574; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,Bpxh,cqyE,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 574
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpyE,cqxh,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 576
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,Bpxh,cqyE,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 598
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpyE,cqxh,drJl,Cszi,etHF,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 600
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,Bpxh,cqyE,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 694
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anIk,bowg,BpyE,cqxh,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 696
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,Bpxh,cqyE,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 718
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anGj,bowg,BpyE,cqxh,drJl,CsHF,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 720
    )

    # Orbit 77: representative state Wick ID 1922; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpyE,cqIk,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1922
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpIk,cqyE,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1925
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpyE,cqIk,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1946
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpIk,cqyE,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1949
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpyE,cqIk,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2042
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpIk,cqyE,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2045
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpyE,cqIk,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2066
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpIk,cqyE,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2069
    )

    # Orbit 78: representative state Wick ID 1921; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpyE,cqGj,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1921
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpGj,cqyE,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1923
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpyE,cqGj,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1945
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpGj,cqyE,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1947
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpyE,cqGj,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2041
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpGj,cqyE,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2043
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpyE,cqGj,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2065
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpGj,cqyE,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2067
    )

    # Orbit 79: representative state Wick ID 1924; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpGj,cqIk,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1924
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpIk,cqGj,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1926
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpGj,cqIk,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1948
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpIk,cqGj,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1950
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpGj,cqIk,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2044
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anxh,boHF,BpIk,cqGj,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2046
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpGj,cqIk,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2068
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anvD,boHF,BpIk,cqGj,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2070
    )

    # Orbit 80: representative state Wick ID 1928; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,boHF,Bpxh,cqIk,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1928
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,boHF,BpIk,cqxh,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1931
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,boHF,Bpxh,cqIk,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1970
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,boHF,BpIk,cqxh,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1973
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,boHF,Bpxh,cqIk,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2048
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,boHF,BpIk,cqxh,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2051
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,boHF,Bpxh,cqIk,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2090
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,boHF,BpIk,cqxh,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2093
    )

    # Orbit 81: representative state Wick ID 1927; weight 2.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,boHF,Bpxh,cqGj,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1927
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,boHF,BpGj,cqxh,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1929
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,boHF,Bpxh,cqGj,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1969
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,boHF,BpGj,cqxh,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1971
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,boHF,Bpxh,cqGj,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2047
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anyE,boHF,BpGj,cqxh,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2049
        +contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,boHF,Bpxh,cqGj,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2089
        -contract(
            'ab,cd,ef,gh,ij,kl,AmyE,anvD,boHF,BpGj,cqxh,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2091
    )

    # Orbit 82: representative state Wick ID 1933; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,Bpxh,cqyE,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1933
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,BpyE,cqxh,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1935
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,Bpxh,cqyE,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1993
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,BpyE,cqxh,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1995
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,Bpxh,cqyE,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2053
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,BpyE,cqxh,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2055
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,Bpxh,cqyE,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2113
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,BpyE,cqxh,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2115
    )

    # Orbit 83: representative state Wick ID 1934; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,Bpxh,cqIk,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1934
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,BpIk,cqxh,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1937
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,Bpxh,cqIk,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1994
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,BpIk,cqxh,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1997
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,Bpxh,cqIk,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2054
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,BpIk,cqxh,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2057
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,Bpxh,cqIk,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2114
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,BpIk,cqxh,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2117
    )

    # Orbit 84: representative state Wick ID 1936; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,BpyE,cqIk,drJl,Cswg,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1936
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,BpIk,cqyE,drJl,Cswg,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1938
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,BpyE,cqIk,drJl,Cswg,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1996
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,BpIk,cqyE,drJl,Cswg,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1998
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,BpyE,cqIk,drJl,Cszi,etwg,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2056
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anGj,boHF,BpIk,cqyE,drJl,Cszi,etwg,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2058
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,BpyE,cqIk,drJl,Cszi,etwg,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2116
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anvD,boHF,BpIk,cqyE,drJl,Cszi,etwg,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2118
    )

    # Orbit 85: representative state Wick ID 1939; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,Bpxh,cqyE,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1939
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,BpyE,cqxh,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1941
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,Bpxh,cqyE,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2017
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,BpyE,cqxh,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2019
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,Bpxh,cqyE,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2059
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,BpyE,cqxh,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2061
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,Bpxh,cqyE,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2137
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,BpyE,cqxh,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2139
    )

    # Orbit 86: representative state Wick ID 1940; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,Bpxh,cqGj,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1940
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,BpGj,cqxh,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1943
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,Bpxh,cqGj,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2018
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,BpGj,cqxh,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2021
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,Bpxh,cqGj,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2060
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,BpGj,cqxh,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2063
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,Bpxh,cqGj,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2138
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,BpGj,cqxh,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2141
    )

    # Orbit 87: representative state Wick ID 1942; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,BpyE,cqGj,drJl,Cswg,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1942
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,BpGj,cqyE,drJl,Cswg,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1944
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,BpyE,cqGj,drJl,Cswg,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2020
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,BpGj,cqyE,drJl,Cswg,etzi,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2022
        -contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,BpyE,cqGj,drJl,Cszi,etwg,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2062
        +contract(
            'ab,cd,ef,gh,ij,kl,AmvD,anIk,boHF,BpGj,cqyE,drJl,Cszi,etwg,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2064
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,BpyE,cqGj,drJl,Cszi,etwg,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2140
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anvD,boHF,BpGj,cqyE,drJl,Cszi,etwg,fuxh,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2142
    )

    # Orbit 88: representative state Wick ID 1957; weight 2.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,boHF,BpvD,cqyE,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1957
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,boHF,BpyE,cqvD,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1959
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,boHF,BpvD,cqyE,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1999
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,boHF,BpyE,cqvD,drJl,Cswg,etzi,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2001
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,boHF,BpvD,cqyE,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2077
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,boHF,BpyE,cqvD,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2079
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,boHF,BpvD,cqyE,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2119
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,boHF,BpyE,cqvD,drJl,Cszi,etwg,fuIk,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2121
    )

    # Orbit 89: representative state Wick ID 1958; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,boHF,BpvD,cqIk,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1958
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,boHF,BpIk,cqvD,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1961
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,boHF,BpvD,cqIk,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2000
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,boHF,BpIk,cqvD,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2003
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,boHF,BpvD,cqIk,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2078
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anGj,boHF,BpIk,cqvD,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2081
        -contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,boHF,BpvD,cqIk,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2120
        +contract(
            'ab,cd,ef,gh,ij,kl,AmGj,anxh,boHF,BpIk,cqvD,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2123
    )

    # Orbit 90: representative state Wick ID 1963; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpvD,cqyE,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1963
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpyE,cqvD,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1965
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpvD,cqyE,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2023
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpyE,cqvD,drJl,Cswg,etzi,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2025
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpvD,cqyE,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2083
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpyE,cqvD,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2085
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpvD,cqyE,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2143
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpyE,cqvD,drJl,Cszi,etwg,fuGj,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2145
    )

    # Orbit 91: representative state Wick ID 1964; weight 4.
    block_values.append(
        
        contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpvD,cqGj,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1964
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpGj,cqvD,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1967
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpvD,cqGj,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2024
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpGj,cqvD,drJl,Cswg,etzi,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2027
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpvD,cqGj,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2084
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpGj,cqvD,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2087
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpvD,cqGj,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2144
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpGj,cqvD,drJl,Cszi,etwg,fuyE,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2147
    )

    # Orbit 92: representative state Wick ID 1966; weight 4.
    block_values.append(
        
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpyE,cqGj,drJl,Cswg,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1966
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpGj,cqyE,drJl,Cswg,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 1968
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpyE,cqGj,drJl,Cswg,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2026
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpGj,cqyE,drJl,Cswg,etzi,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2028
        +contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpyE,cqGj,drJl,Cszi,etwg,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2086
        -contract(
            'ab,cd,ef,gh,ij,kl,Amxh,anIk,boHF,BpGj,cqyE,drJl,Cszi,etwg,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2088
        -contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpyE,cqGj,drJl,Cszi,etwg,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2146
        +contract(
            'ab,cd,ef,gh,ij,kl,AmIk,anxh,boHF,BpGj,cqyE,drJl,Cszi,etwg,fuvD,mno,pqr,stu,vwx,yzG,HIJ,ABCDEF->',
            *projected_operands,
            optimize="greedy",
        )  # Wick ID 2148
    )

    block_weights = (
        2, 4, 4, 4, 4, 4, 2, 4, 4, 4, 2, 4, 4, 4, 4, 2,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4,
    )
    return sum(
        (weight * block for weight, block in zip(block_weights, block_values)),
        0j,
    )
